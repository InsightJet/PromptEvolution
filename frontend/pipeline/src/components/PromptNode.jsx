import React, { memo } from 'react';
import { Handle, Position } from '@xyflow/react';

function PromptNode({ data, selected }) {
  const inputs = data.inputVariables || [];
  const hasOutput = !!data.outputVariable;

  return (
    <div className={`prompt-node ${selected ? 'selected' : ''} ${data.evolving ? 'evolving' : ''}`}>
      <Handle type="target" position={Position.Left} className="node-handle handle-input" />

      <div className="prompt-node-header">
        <span className="prompt-node-label">{data.label || 'Untitled Node'}</span>
      </div>

      {/* Input variables flowing in */}
      {inputs.length > 0 && (
        <div className="prompt-node-vars-in">
          {inputs.map((v) => (
            <span key={v} className="prompt-node-var in">{v} &rarr;</span>
          ))}
        </div>
      )}

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

      {/* Output: LLM response becomes this variable */}
      <div className="prompt-node-footer">
        <span className="prompt-node-llm-tag">LLM</span>
        <span className="prompt-node-flow-icon">&rarr;</span>
        {hasOutput ? (
          <span className="prompt-node-var out">{data.outputVariable}</span>
        ) : (
          <span className="prompt-node-var out empty">name output</span>
        )}
      </div>

      <Handle type="source" position={Position.Right} className="node-handle handle-output" />
    </div>
  );
}

export default memo(PromptNode);
