import React, { useState, useCallback, useRef } from 'react';
import { ReactFlowProvider } from '@xyflow/react';
import PipelineCanvas from './components/PipelineCanvas';
import NodeEditor from './components/NodeEditor';
import TestPanel from './components/TestPanel';
import Toolbar from './components/Toolbar';
import EvolutionOverlay from './components/EvolutionOverlay';
import ResultsOverlay from './components/ResultsOverlay';
import { usePipeline } from './hooks/usePipeline';
import { useEvolution } from './hooks/useEvolution';

export default function App() {
  const pipeline = usePipeline();
  const evolution = useEvolution(pipeline);
  const [executionResult, setExecutionResult] = useState(null);

  const handleRunPipeline = useCallback(async () => {
    if (pipeline.nodes.length === 0) return;
    const testInputs = pipeline.testInputs;
    if (!testInputs.length) {
      alert('Add at least one test input');
      return;
    }
    if (!pipeline.taskModel.api_key) {
      alert('Please configure a task model with API key');
      return;
    }

    try {
      const pipelineData = pipeline.toPipelineJSON();
      const res = await fetch('/api/pipeline/execute', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          ...(getAuthToken() ? { Authorization: `Bearer ${getAuthToken()}` } : {}),
        },
        body: JSON.stringify({
          pipeline: pipelineData,
          test_input: testInputs[0],
          task_model: pipeline.taskModel,
        }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || 'Execution failed');
      setExecutionResult(data);
    } catch (err) {
      alert('Pipeline execution failed: ' + err.message);
    }
  }, [pipeline]);

  return (
    <ReactFlowProvider>
      <div className="pl-container">
        <Toolbar
          pipeline={pipeline}
          onRun={handleRunPipeline}
          onEvolve={() => evolution.start(pipeline)}
          isRunning={evolution.isRunning}
        />

        <div className="pl-workspace">
          <NodeEditor
            node={pipeline.selectedNode}
            allNodes={pipeline.nodes}
            onUpdate={pipeline.updateNode}
            onDelete={pipeline.deleteSelectedNode}
          />

          <PipelineCanvas
            nodes={pipeline.flowNodes}
            edges={pipeline.flowEdges}
            onNodesChange={pipeline.onNodesChange}
            onEdgesChange={pipeline.onEdgesChange}
            onConnect={pipeline.onConnect}
            onNodeClick={pipeline.selectNode}
            evolvingNodeId={evolution.currentNodeEvolving}
          />
        </div>

        <TestPanel
          pipeline={pipeline}
          executionResult={executionResult}
        />

        {evolution.isRunning && (
          <EvolutionOverlay
            evolution={evolution}
            onStop={() => evolution.stop()}
          />
        )}

        {evolution.isComplete && (
          <ResultsOverlay
            evolution={evolution}
            pipeline={pipeline}
            onClose={() => evolution.reset()}
            onApply={() => {
              pipeline.applyEvolvedPipeline(evolution.evolvedPipeline);
              evolution.reset();
            }}
          />
        )}
      </div>
    </ReactFlowProvider>
  );
}

function getAuthToken() {
  return localStorage.getItem('authToken');
}
