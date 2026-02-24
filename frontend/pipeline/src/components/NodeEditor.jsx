import React, { useRef, useEffect, useMemo, useCallback } from 'react';

export default function NodeEditor({ node, allNodes, onUpdate, onDelete }) {
  const textareaRef = useRef(null);

  useEffect(() => {
    if (node && textareaRef.current) {
      textareaRef.current.focus();
    }
  }, [node?.id]);

  // Compute available variables from OTHER nodes' outputs
  const availableVars = useMemo(() => {
    if (!node || !allNodes) return [];
    return allNodes
      .filter((n) => n.id !== node.id && n.data.outputVariable)
      .map((n) => ({
        nodeId: n.id,
        label: n.data.label || n.id,
        variable: n.data.outputVariable,
      }));
  }, [node?.id, allNodes]);

  // Insert variable at cursor position in textarea
  const insertVariable = useCallback((varName) => {
    if (!textareaRef.current || !node) return;
    const ta = textareaRef.current;
    const start = ta.selectionStart;
    const end = ta.selectionEnd;
    const text = ta.value;
    const insertion = `{{${varName}}}`;
    const newText = text.substring(0, start) + insertion + text.substring(end);
    onUpdate(node.id, { promptTemplate: newText });

    // Restore cursor after insertion
    setTimeout(() => {
      ta.focus();
      ta.selectionStart = ta.selectionEnd = start + insertion.length;
    }, 0);
  }, [node, onUpdate]);

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

  // Check which available vars are already used in the template
  const usedVars = new Set(data.inputVariables || []);

  return (
    <div className="pl-node-panel pl-node-panel-active">
      <div className="pl-editor-node-indicator">
        <span className="pl-editor-node-dot" />
        <span className="pl-editor-node-name">{data.label || 'Untitled Node'}</span>
      </div>

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

      {/* Available Variables from other nodes */}
      {availableVars.length > 0 && (
        <div className="pl-form-group">
          <label className="pl-label">
            Available Variables
            <span className="pl-hint-inline">click to insert</span>
          </label>
          <div className="pl-available-vars">
            {availableVars.map((av) => (
              <button
                key={av.nodeId}
                className={`pl-available-var ${usedVars.has(av.variable) ? 'used' : ''}`}
                onClick={() => insertVariable(av.variable)}
                title={`From "${av.label}" — click to insert {{${av.variable}}}`}
              >
                <span className="pl-av-name">{`{{${av.variable}}}`}</span>
                <span className="pl-av-source">{av.label}</span>
              </button>
            ))}
          </div>
        </div>
      )}

      <div className="pl-form-group">
        <label className="pl-label">Prompt Template</label>
        <textarea
          ref={textareaRef}
          className="pl-textarea"
          value={data.promptTemplate}
          onChange={(e) => onUpdate(node.id, { promptTemplate: e.target.value })}
          placeholder={availableVars.length > 0
            ? 'Click an available variable above to insert it, or type {{variable_name}} manually'
            : 'Type your prompt here.\nThis is the first node — its output feeds into downstream nodes.'}
          rows={8}
        />
      </div>

      <div className="pl-form-group">
        <label className="pl-label">
          Mapped Inputs
          <span className="pl-hint-inline">auto-detected from template</span>
        </label>
        <div className="pl-var-list">
          {data.inputVariables && data.inputVariables.length > 0 ? (
            data.inputVariables.map((v) => {
              const source = availableVars.find((av) => av.variable === v);
              return (
                <div key={v} className="pl-mapped-var">
                  <span className="pl-var-badge input">{`{{${v}}}`}</span>
                  {source ? (
                    <span className="pl-var-arrow-source">
                      <span className="pl-var-arrow">&larr;</span>
                      <span className="pl-var-source-label">{source.label}</span>
                    </span>
                  ) : (
                    <span className="pl-var-arrow-source">
                      <span className="pl-var-arrow">&larr;</span>
                      <span className="pl-var-source-label unresolved">pipeline input</span>
                    </span>
                  )}
                </div>
              );
            })
          ) : (
            <span className="pl-var-empty">No variables detected</span>
          )}
        </div>
      </div>

      {/* Data Flow Diagram */}
      <div className="pl-flow-diagram">
        <div className="pl-flow-step">
          <span className="pl-flow-label">Prompt sent to</span>
          <span className="pl-flow-badge llm">LLM</span>
        </div>
        <div className="pl-flow-arrow-down">&darr;</div>
        <div className="pl-flow-step">
          <span className="pl-flow-label">Full response saved as</span>
          <span className="pl-flow-badge out">{data.outputVariable || '???'}</span>
        </div>
      </div>

      <div className="pl-form-group">
        <label className="pl-label">Output Variable Name</label>
        <input
          type="text"
          className="pl-input"
          value={data.outputVariable}
          onChange={(e) => onUpdate(node.id, { outputVariable: e.target.value })}
          placeholder="e.g., image_descriptions, summary, analysis"
        />
        <span className="pl-hint-text">
          The LLM's entire response is stored as this variable.
          Other nodes use it via {`{{${data.outputVariable || 'variable_name'}}}`}
        </span>
      </div>

      <button className="pl-btn pl-btn-danger" onClick={onDelete}>
        Delete Node
      </button>
    </div>
  );
}
