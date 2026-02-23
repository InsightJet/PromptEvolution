import React, { memo } from 'react';
import { Handle, Position } from '@xyflow/react';

function PromptNode({ data, selected }) {
  return (
    <div className={`prompt-node ${selected ? 'selected' : ''} ${data.evolving ? 'evolving' : ''}`}>
      <Handle type="target" position={Position.Left} className="node-handle handle-input" />

      <div className="prompt-node-header">
        <span className="prompt-node-label">{data.label || 'Untitled Node'}</span>
      </div>

      <div className="prompt-node-body">
        {data.promptTemplate ? (
          <span className="prompt-node-preview">
            {data.promptTemplate.substring(0, 60)}
            {data.promptTemplate.length > 60 ? '...' : ''}
          </span>
        ) : (
          <span className="prompt-node-empty">No prompt yet</span>
        )}
      </div>

      <div className="prompt-node-footer">
        {data.outputVariable ? (
          <span className="prompt-node-output">
            &rarr; {data.outputVariable}
          </span>
        ) : (
          <span className="prompt-node-output empty">&rarr; set output var</span>
        )}
      </div>

      <Handle type="source" position={Position.Right} className="node-handle handle-output" />
    </div>
  );
}

export default memo(PromptNode);
