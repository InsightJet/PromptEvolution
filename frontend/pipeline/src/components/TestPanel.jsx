import React, { useState } from 'react';

export default function TestPanel({ pipeline, executionResult }) {
  const [collapsed, setCollapsed] = useState(false);

  // Compute pipeline inputs (variables not produced by any node)
  const pipelineInputVars = (() => {
    const allOutputs = new Set(pipeline.nodes.map((n) => n.data.outputVariable).filter(Boolean));
    const allInputs = new Set();
    pipeline.nodes.forEach((n) => (n.data.inputVariables || []).forEach((v) => allInputs.add(v)));
    return [...allInputs].filter((v) => !allOutputs.has(v));
  })();

  const addTestInput = () => {
    const newInput = {};
    pipelineInputVars.forEach((v) => (newInput[v] = ''));
    pipeline.setTestInputs([...pipeline.testInputs, newInput]);
  };

  const updateTestInput = (index, key, value) => {
    const updated = [...pipeline.testInputs];
    updated[index] = { ...updated[index], [key]: value };
    pipeline.setTestInputs(updated);
  };

  const removeTestInput = (index) => {
    pipeline.setTestInputs(pipeline.testInputs.filter((_, i) => i !== index));
  };

  return (
    <div className={`pl-test-panel ${collapsed ? 'collapsed' : ''}`}>
      <div className="pl-test-header" onClick={() => setCollapsed(!collapsed)}>
        <h4>Configuration</h4>
        <span className="pl-collapse-icon">{collapsed ? '\u25B6' : '\u25BC'}</span>
      </div>

      {!collapsed && (
        <div className="pl-test-content">
          {/* Model Config */}
          <div className="pl-config-row">
            <div className="pl-config-section">
              <label className="pl-label">Task Model</label>
              <div className="pl-model-row">
                <select
                  className="pl-select pl-select-sm"
                  value={pipeline.taskModel.provider}
                  onChange={(e) => pipeline.setTaskModel({ ...pipeline.taskModel, provider: e.target.value })}
                >
                  <option value="openai">OpenAI</option>
                  <option value="anthropic">Anthropic</option>
                  <option value="google">Google</option>
                  <option value="groq">Groq</option>
                  <option value="mistral">Mistral</option>
                </select>
                <input
                  type="text"
                  className="pl-input pl-input-sm"
                  value={pipeline.taskModel.model}
                  onChange={(e) => pipeline.setTaskModel({ ...pipeline.taskModel, model: e.target.value })}
                  placeholder="Model name"
                />
                <input
                  type="password"
                  className="pl-input pl-input-sm"
                  value={pipeline.taskModel.api_key}
                  onChange={(e) => pipeline.setTaskModel({ ...pipeline.taskModel, api_key: e.target.value })}
                  placeholder="API Key"
                />
              </div>
            </div>
            <div className="pl-config-section">
              <label className="pl-label">Judge Model</label>
              <div className="pl-model-row">
                <select
                  className="pl-select pl-select-sm"
                  value={pipeline.judgeModel.provider}
                  onChange={(e) => pipeline.setJudgeModel({ ...pipeline.judgeModel, provider: e.target.value })}
                >
                  <option value="openai">OpenAI</option>
                  <option value="anthropic">Anthropic</option>
                  <option value="google">Google</option>
                  <option value="groq">Groq</option>
                  <option value="mistral">Mistral</option>
                </select>
                <input
                  type="text"
                  className="pl-input pl-input-sm"
                  value={pipeline.judgeModel.model}
                  onChange={(e) => pipeline.setJudgeModel({ ...pipeline.judgeModel, model: e.target.value })}
                  placeholder="Model name"
                />
                <input
                  type="password"
                  className="pl-input pl-input-sm"
                  value={pipeline.judgeModel.api_key}
                  onChange={(e) => pipeline.setJudgeModel({ ...pipeline.judgeModel, api_key: e.target.value })}
                  placeholder="API Key"
                />
              </div>
            </div>
          </div>

          {/* Judge Prompt */}
          <div className="pl-form-group">
            <label className="pl-label">Judge Prompt</label>
            <textarea
              className="pl-textarea pl-textarea-sm"
              value={pipeline.judgePrompt}
              onChange={(e) => pipeline.setJudgePrompt(e.target.value)}
              placeholder="How to evaluate the final pipeline output...&#10;&#10;End with:&#10;SCORE: [0-100]&#10;FEEDBACK: [details]"
              rows={3}
            />
          </div>

          {/* Test Inputs */}
          <div className="pl-form-group">
            <div className="pl-test-inputs-header">
              <label className="pl-label">
                Test Inputs
                {pipelineInputVars.length > 0 && (
                  <span className="pl-hint-inline">
                    Pipeline expects: {pipelineInputVars.map((v) => `{{${v}}}`).join(', ')}
                  </span>
                )}
              </label>
              <button className="pl-btn pl-btn-sm pl-btn-secondary" onClick={addTestInput}>
                + Add
              </button>
            </div>

            {pipeline.testInputs.map((input, idx) => (
              <div key={idx} className="pl-test-input-card">
                <div className="pl-test-input-header">
                  <span className="pl-test-input-label">Test {idx + 1}</span>
                  <button
                    className="pl-btn-icon"
                    onClick={() => removeTestInput(idx)}
                    title="Remove"
                  >
                    &times;
                  </button>
                </div>
                {pipelineInputVars.map((varName) => (
                  <div key={varName} className="pl-test-var">
                    <label className="pl-label-sm">{varName}</label>
                    <textarea
                      className="pl-textarea pl-textarea-xs"
                      value={input[varName] || ''}
                      onChange={(e) => updateTestInput(idx, varName, e.target.value)}
                      placeholder={`Value for {{${varName}}}`}
                      rows={2}
                    />
                  </div>
                ))}
              </div>
            ))}
          </div>

          {/* Execution Result */}
          {executionResult && (
            <div className="pl-execution-result">
              <h4 className="pl-panel-title">Execution Result</h4>
              <div className="pl-result-output">
                <label className="pl-label">Final Output</label>
                <pre className="pl-pre">{executionResult.final_output}</pre>
              </div>
              {executionResult.intermediate_outputs && (
                <details className="pl-details">
                  <summary>Intermediate Outputs</summary>
                  {Object.entries(executionResult.intermediate_outputs).map(([nodeId, output]) => (
                    <div key={nodeId} className="pl-intermediate">
                      <label className="pl-label-sm">{nodeId}</label>
                      <pre className="pl-pre-sm">{output}</pre>
                    </div>
                  ))}
                </details>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
