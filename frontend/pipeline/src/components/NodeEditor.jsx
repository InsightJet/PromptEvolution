import React from 'react';

export default function NodeEditor({ node, onUpdate, onDelete }) {
  if (!node) {
    return (
      <div className="pl-node-panel">
        <div className="pl-node-placeholder">
          <p>Select a node on the canvas to edit it</p>
          <p className="pl-hint">Or click "+ Add Node" in the toolbar</p>
        </div>
      </div>
    );
  }

  const { data } = node;

  return (
    <div className="pl-node-panel">
      <h4 className="pl-panel-title">Node Settings</h4>

      <div className="pl-form-group">
        <label className="pl-label">Label</label>
        <input
          type="text"
          className="pl-input"
          value={data.label}
          onChange={(e) => onUpdate(node.id, { label: e.target.value })}
          placeholder="Node name"
        />
      </div>

      <div className="pl-form-group">
        <label className="pl-label">Prompt Template</label>
        <textarea
          className="pl-textarea"
          value={data.promptTemplate}
          onChange={(e) => onUpdate(node.id, { promptTemplate: e.target.value })}
          placeholder={'Use {{variable}} for inputs from other nodes\n\nExample:\nReview these images:\n{{image_descriptions}}\n\nProvide feedback on quality...'}
          rows={8}
        />
      </div>

      <div className="pl-form-group">
        <label className="pl-label">
          Input Variables
          <span className="pl-hint-inline">auto-detected from template</span>
        </label>
        <div className="pl-var-list">
          {data.inputVariables && data.inputVariables.length > 0 ? (
            data.inputVariables.map((v) => (
              <span key={v} className="pl-var-badge input">
                {`{{${v}}}`}
              </span>
            ))
          ) : (
            <span className="pl-var-empty">No variables detected</span>
          )}
        </div>
      </div>

      <div className="pl-form-group">
        <label className="pl-label">Output Variable Name</label>
        <input
          type="text"
          className="pl-input"
          value={data.outputVariable}
          onChange={(e) => onUpdate(node.id, { outputVariable: e.target.value })}
          placeholder="e.g., entities, summary, final_output"
        />
        <span className="pl-hint-text">
          Other nodes can use this as {`{{${data.outputVariable || 'variable_name'}}}`}
        </span>
      </div>

      <button className="pl-btn pl-btn-danger" onClick={onDelete}>
        Delete Node
      </button>
    </div>
  );
}
