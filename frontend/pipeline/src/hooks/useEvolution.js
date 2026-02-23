import { useState, useCallback, useRef } from 'react';

function getAuthToken() {
  return localStorage.getItem('authToken');
}

export function useEvolution() {
  const [isRunning, setIsRunning] = useState(false);
  const [isComplete, setIsComplete] = useState(false);
  const [initialScore, setInitialScore] = useState(null);
  const [bestScore, setBestScore] = useState(null);
  const [currentRound, setCurrentRound] = useState(0);
  const [currentNodeEvolving, setCurrentNodeEvolving] = useState(null);
  const [nodeEvolutionLog, setNodeEvolutionLog] = useState([]);
  const [evolvedPipeline, setEvolvedPipeline] = useState(null);
  const [logs, setLogs] = useState([]);
  const [status, setStatus] = useState('idle');
  const eventSourceRef = useRef(null);

  const start = useCallback(async (pipeline) => {
    const pipelineData = pipeline.toPipelineJSON();

    if (!pipelineData.nodes.length) {
      alert('Add at least one node to the pipeline');
      return;
    }
    if (!pipelineData.judge_prompt) {
      alert('Please add a judge prompt');
      return;
    }
    if (!pipelineData.test_inputs?.length) {
      alert('Please add at least one test input');
      return;
    }

    // Save pipeline first
    const headers = {
      'Content-Type': 'application/json',
      ...(getAuthToken() ? { Authorization: `Bearer ${getAuthToken()}` } : {}),
    };

    try {
      const saveRes = await fetch(
        pipeline.pipelineId
          ? `/api/pipeline/save?pipeline_id=${pipeline.pipelineId}`
          : '/api/pipeline/save',
        { method: 'POST', headers, body: JSON.stringify(pipelineData) }
      );
      const saveData = await saveRes.json();
      if (!saveRes.ok) throw new Error(saveData.detail || 'Save failed');

      const savedId = saveData.id;
      pipeline.setPipelineId(savedId);

      // Start evolution
      setIsRunning(true);
      setIsComplete(false);
      setLogs([]);
      setNodeEvolutionLog([]);
      setInitialScore(null);
      setBestScore(null);
      setCurrentRound(0);
      setCurrentNodeEvolving(null);
      setEvolvedPipeline(null);
      setStatus('running');

      const evoRes = await fetch('/api/pipeline/evolution/start', {
        method: 'POST',
        headers,
        body: JSON.stringify({
          pipeline_id: savedId,
          task_model: pipelineData.task_model,
          judge_model: pipelineData.judge_model,
          max_iterations: 5,
          max_rounds: 3,
        }),
      });
      const evoData = await evoRes.json();
      if (!evoRes.ok) throw new Error(evoData.detail || 'Evolution start failed');

      // Connect SSE
      const token = getAuthToken();
      const url = `/api/pipeline/evolution/${evoData.session_id}/stream${token ? `?token=${encodeURIComponent(token)}` : ''}`;
      const es = new EventSource(url);
      eventSourceRef.current = es;

      es.addEventListener('log', (event) => {
        const entry = JSON.parse(event.data);
        setLogs((prev) => [...prev, entry]);
      });

      es.addEventListener('status', (event) => {
        const s = JSON.parse(event.data);
        if (s.initial_score != null) setInitialScore(s.initial_score);
        if (s.best_score != null) setBestScore(s.best_score);
        if (s.current_round != null) setCurrentRound(s.current_round);
        if (s.current_node_evolving !== undefined) setCurrentNodeEvolving(s.current_node_evolving);
        if (s.node_evolution_log) setNodeEvolutionLog(s.node_evolution_log);
        if (s.evolved_pipeline) setEvolvedPipeline(s.evolved_pipeline);

        if (s.status === 'completed' || s.status === 'error' || s.status === 'stopped') {
          setIsRunning(false);
          setIsComplete(true);
          setStatus(s.status);
          es.close();
        }
      });

      es.onerror = () => {
        setIsRunning(false);
        es.close();
      };
    } catch (err) {
      setIsRunning(false);
      alert('Evolution failed: ' + err.message);
    }
  }, []);

  const stop = useCallback(async () => {
    if (eventSourceRef.current) {
      eventSourceRef.current.close();
    }
    setIsRunning(false);
    setStatus('stopped');
  }, []);

  const reset = useCallback(() => {
    setIsComplete(false);
    setIsRunning(false);
    setStatus('idle');
    setLogs([]);
  }, []);

  return {
    isRunning,
    isComplete,
    initialScore,
    bestScore,
    currentRound,
    currentNodeEvolving,
    nodeEvolutionLog,
    evolvedPipeline,
    logs,
    status,
    start,
    stop,
    reset,
  };
}
