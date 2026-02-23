import React, { useMemo } from 'react';
import { ReactFlow, Background, Controls, MiniMap } from '@xyflow/react';
import '@xyflow/react/dist/style.css';
import PromptNode from './PromptNode';

const nodeTypes = { promptNode: PromptNode };

export default function PipelineCanvas({
  nodes,
  edges,
  onNodesChange,
  onEdgesChange,
  onConnect,
  onNodeClick,
  evolvingNodeId,
}) {
  // Mark evolving node
  const processedNodes = useMemo(() => {
    return nodes.map((n) => ({
      ...n,
      data: {
        ...n.data,
        evolving: n.id === evolvingNodeId,
      },
    }));
  }, [nodes, evolvingNodeId]);

  return (
    <div className="pl-canvas-wrapper">
      <ReactFlow
        nodes={processedNodes}
        edges={edges}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        onConnect={onConnect}
        onNodeClick={onNodeClick}
        nodeTypes={nodeTypes}
        fitView
        proOptions={{ hideAttribution: true }}
        defaultEdgeOptions={{
          animated: true,
          style: { stroke: 'rgba(0, 212, 255, 0.5)', strokeWidth: 2 },
        }}
      >
        <Background color="rgba(255,255,255,0.03)" gap={20} />
        <Controls
          showInteractive={false}
          style={{ background: 'rgba(0,0,0,0.4)', borderRadius: 8, border: '1px solid rgba(255,255,255,0.1)' }}
        />
        <MiniMap
          nodeStrokeColor="rgba(0,212,255,0.3)"
          nodeColor="rgba(0,212,255,0.1)"
          maskColor="rgba(0,0,0,0.7)"
          style={{ background: 'rgba(0,0,0,0.3)', borderRadius: 8, border: '1px solid rgba(255,255,255,0.1)' }}
        />
      </ReactFlow>
    </div>
  );
}
