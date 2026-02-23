import { useState, useCallback, useMemo } from 'react';
import { useNodesState, useEdgesState, addEdge } from '@xyflow/react';

let nextNodeId = 1;

export function usePipeline() {
  const [nodes, setNodes, onNodesChange] = useNodesState([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState([]);
  const [selectedNodeId, setSelectedNodeId] = useState(null);
  const [pipelineId, setPipelineId] = useState(null);
  const [pipelineName, setPipelineName] = useState('');
  const [judgePrompt, setJudgePrompt] = useState('');
  const [testInputs, setTestInputs] = useState([]);
  const [taskModel, setTaskModel] = useState({ provider: 'openai', model: 'gpt-4o', api_key: '' });
  const [judgeModel, setJudgeModel] = useState({ provider: 'openai', model: 'gpt-4o', api_key: '' });

  const addNode = useCallback(() => {
    const id = `node_${nextNodeId++}`;
    const newNode = {
      id,
      type: 'promptNode',
      position: { x: 150 + nodes.length * 280, y: 200 },
      data: {
        label: `Node ${nodes.length + 1}`,
        promptTemplate: '',
        inputVariables: [],
        outputVariable: '',
      },
    };
    setNodes((nds) => [...nds, newNode]);
    setSelectedNodeId(id);
    return id;
  }, [nodes.length, setNodes]);

  const updateNode = useCallback((nodeId, updates) => {
    setNodes((nds) =>
      nds.map((n) => {
        if (n.id !== nodeId) return n;
        const newData = { ...n.data, ...updates };

        // Auto-detect input variables from template
        if (updates.promptTemplate !== undefined) {
          const vars = [];
          const regex = /\{\{(\w+)\}\}/g;
          let match;
          while ((match = regex.exec(updates.promptTemplate)) !== null) {
            if (!vars.includes(match[1])) vars.push(match[1]);
          }
          newData.inputVariables = vars;
        }

        return { ...n, data: newData };
      })
    );

    // Auto-wire edges after update
    if (updates.promptTemplate !== undefined || updates.outputVariable !== undefined) {
      setTimeout(() => autoWireEdges(), 0);
    }
  }, [setNodes]);

  const autoWireEdges = useCallback(() => {
    setNodes((currentNodes) => {
      setEdges(() => {
        const outputMap = {};
        currentNodes.forEach((n) => {
          if (n.data.outputVariable) {
            outputMap[n.data.outputVariable] = n.id;
          }
        });

        const newEdges = [];
        currentNodes.forEach((node) => {
          (node.data.inputVariables || []).forEach((varName) => {
            const sourceId = outputMap[varName];
            if (sourceId && sourceId !== node.id) {
              const edgeId = `${sourceId}-${node.id}-${varName}`;
              newEdges.push({
                id: edgeId,
                source: sourceId,
                target: node.id,
                label: varName,
                animated: true,
                style: { stroke: 'rgba(0, 212, 255, 0.5)' },
                labelStyle: { fill: 'rgba(0, 212, 255, 0.8)', fontSize: 11 },
              });
            }
          });
        });

        return newEdges;
      });
      return currentNodes; // Don't modify nodes
    });
  }, [setNodes, setEdges]);

  const deleteSelectedNode = useCallback(() => {
    if (!selectedNodeId) return;
    setNodes((nds) => nds.filter((n) => n.id !== selectedNodeId));
    setEdges((eds) =>
      eds.filter((e) => e.source !== selectedNodeId && e.target !== selectedNodeId)
    );
    setSelectedNodeId(null);
  }, [selectedNodeId, setNodes, setEdges]);

  const selectNode = useCallback((_event, node) => {
    setSelectedNodeId(node.id);
  }, []);

  const onConnect = useCallback(
    (params) => {
      setEdges((eds) =>
        addEdge(
          {
            ...params,
            animated: true,
            style: { stroke: 'rgba(0, 212, 255, 0.5)' },
          },
          eds
        )
      );
    },
    [setEdges]
  );

  const selectedNode = useMemo(() => {
    return nodes.find((n) => n.id === selectedNodeId) || null;
  }, [nodes, selectedNodeId]);

  // Convert internal state to pipeline JSON for API
  const toPipelineJSON = useCallback(() => {
    const allOutputs = new Set(nodes.map((n) => n.data.outputVariable).filter(Boolean));
    const allInputs = new Set();
    nodes.forEach((n) => (n.data.inputVariables || []).forEach((v) => allInputs.add(v)));
    const pipelineInputs = [...allInputs].filter((v) => !allOutputs.has(v));

    // Determine pipeline output (last node's output variable)
    const lastNode = nodes[nodes.length - 1];
    const pipelineOutput = lastNode?.data.outputVariable || '';

    return {
      name: pipelineName || 'Untitled Pipeline',
      nodes: nodes.map((n) => ({
        id: n.id,
        label: n.data.label,
        prompt_template: n.data.promptTemplate,
        input_variables: n.data.inputVariables || [],
        output_variable: n.data.outputVariable || '',
        position: n.position,
      })),
      edges: edges.map((e) => ({
        from_node: e.source,
        from_var: e.label || '',
        to_node: e.target,
        to_var: e.label || '',
      })),
      pipeline_inputs: pipelineInputs,
      pipeline_output: pipelineOutput,
      judge_prompt: judgePrompt,
      test_inputs: testInputs,
      task_model: taskModel,
      judge_model: judgeModel,
    };
  }, [nodes, edges, pipelineName, judgePrompt, testInputs, taskModel, judgeModel]);

  // Load pipeline from API response
  const loadPipeline = useCallback((data) => {
    const pj = data.pipeline_json || data;
    setPipelineId(data.id || null);
    setPipelineName(pj.name || data.name || '');
    setJudgePrompt(data.judge_prompt || pj.judge_prompt || '');
    setTestInputs(data.test_inputs || pj.test_inputs || []);

    if (data.task_model_config) setTaskModel(data.task_model_config);
    if (data.judge_model_config) setJudgeModel(data.judge_model_config);

    const loadedNodes = (pj.nodes || []).map((n) => ({
      id: n.id,
      type: 'promptNode',
      position: n.position || { x: 0, y: 0 },
      data: {
        label: n.label,
        promptTemplate: n.prompt_template,
        inputVariables: n.input_variables || [],
        outputVariable: n.output_variable || '',
      },
    }));

    // Update nextNodeId
    loadedNodes.forEach((n) => {
      const num = parseInt(n.id.replace('node_', ''));
      if (!isNaN(num) && num >= nextNodeId) nextNodeId = num + 1;
    });

    setNodes(loadedNodes);

    // Build edges from loaded data
    const loadedEdges = (pj.edges || []).map((e) => ({
      id: `${e.from_node}-${e.to_node}-${e.from_var}`,
      source: e.from_node,
      target: e.to_node,
      label: e.from_var,
      animated: true,
      style: { stroke: 'rgba(0, 212, 255, 0.5)' },
      labelStyle: { fill: 'rgba(0, 212, 255, 0.8)', fontSize: 11 },
    }));
    setEdges(loadedEdges);
  }, [setNodes, setEdges]);

  const applyEvolvedPipeline = useCallback((evolvedPipeline) => {
    if (!evolvedPipeline) return;
    // Update node prompts from evolved pipeline
    const evolvedNodes = evolvedPipeline.nodes || [];
    setNodes((nds) =>
      nds.map((n) => {
        const evolved = evolvedNodes.find((en) => en.id === n.id);
        if (evolved) {
          return {
            ...n,
            data: { ...n.data, promptTemplate: evolved.prompt_template },
          };
        }
        return n;
      })
    );
  }, [setNodes]);

  return {
    nodes,
    edges,
    flowNodes: nodes,
    flowEdges: edges,
    selectedNode,
    selectedNodeId,
    pipelineId,
    pipelineName,
    judgePrompt,
    testInputs,
    taskModel,
    judgeModel,
    setPipelineName,
    setJudgePrompt,
    setTestInputs,
    setTaskModel,
    setJudgeModel,
    addNode,
    updateNode,
    deleteSelectedNode,
    selectNode,
    onNodesChange,
    onEdgesChange,
    onConnect,
    toPipelineJSON,
    loadPipeline,
    applyEvolvedPipeline,
    setPipelineId,
  };
}
