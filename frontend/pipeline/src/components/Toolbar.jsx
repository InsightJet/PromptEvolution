import React, { useState, useEffect, useCallback } from 'react';

function getAuthToken() {
  return localStorage.getItem('authToken');
}

export default function Toolbar({ pipeline, onRun, onEvolve, isRunning }) {
  const [savedPipelines, setSavedPipelines] = useState([]);
  const [saving, setSaving] = useState(false);
  const [saveLabel, setSaveLabel] = useState('Save');

  // Load saved pipelines list
  useEffect(() => {
    loadPipelineList();
  }, []);

  const loadPipelineList = async () => {
    try {
      const headers = getAuthToken()
        ? { Authorization: `Bearer ${getAuthToken()}` }
        : {};
      const res = await fetch('/api/pipeline/list', { headers });
      if (res.ok) {
        const data = await res.json();
        setSavedPipelines(data.pipelines || []);
      }
    } catch {
      // ignore
    }
  };

  const handleSave = useCallback(async () => {
    if (!pipeline.pipelineName) {
      alert('Please enter a pipeline name');
      return;
    }
    setSaving(true);
    try {
      const data = pipeline.toPipelineJSON();
      const headers = {
        'Content-Type': 'application/json',
        ...(getAuthToken() ? { Authorization: `Bearer ${getAuthToken()}` } : {}),
      };
      const url = pipeline.pipelineId
        ? `/api/pipeline/save?pipeline_id=${pipeline.pipelineId}`
        : '/api/pipeline/save';
      const res = await fetch(url, {
        method: 'POST',
        headers,
        body: JSON.stringify(data),
      });
      const result = await res.json();
      if (res.ok) {
        pipeline.setPipelineId(result.id);
        setSaveLabel('Saved \u2713');
        setTimeout(() => setSaveLabel('Save'), 1500);
        loadPipelineList();
      } else {
        alert('Save failed: ' + (result.detail || 'Unknown error'));
      }
    } catch (err) {
      alert('Save failed: ' + err.message);
    }
    setSaving(false);
  }, [pipeline]);

  const handleLoad = useCallback(async (pipelineId) => {
    if (!pipelineId) {
      // New pipeline
      pipeline.setPipelineId(null);
      pipeline.setPipelineName('');
      return;
    }
    try {
      const headers = getAuthToken()
        ? { Authorization: `Bearer ${getAuthToken()}` }
        : {};
      const res = await fetch(`/api/pipeline/${pipelineId}`, { headers });
      if (res.ok) {
        const data = await res.json();
        pipeline.loadPipeline(data);
      }
    } catch (err) {
      alert('Load failed: ' + err.message);
    }
  }, [pipeline]);

  return (
    <div className="pl-toolbar">
      <select
        className="pl-select"
        value={pipeline.pipelineId || ''}
        onChange={(e) => handleLoad(e.target.value)}
      >
        <option value="">-- New Pipeline --</option>
        {savedPipelines.map((p) => (
          <option key={p.id} value={p.id}>
            {p.name}
          </option>
        ))}
      </select>

      <input
        type="text"
        className="pl-input pl-name-input"
        value={pipeline.pipelineName}
        onChange={(e) => pipeline.setPipelineName(e.target.value)}
        placeholder="Pipeline name"
      />

      <button className="pl-btn pl-btn-secondary" onClick={pipeline.addNode}>
        + Add Node
      </button>

      <div className="pl-toolbar-spacer" />

      <button
        className={`pl-btn pl-btn-save ${saveLabel.includes('\u2713') ? 'saved' : ''}`}
        onClick={handleSave}
        disabled={saving}
      >
        {saveLabel}
      </button>

      <button className="pl-btn pl-btn-secondary" onClick={onRun} disabled={isRunning}>
        Run Pipeline
      </button>

      <button className="pl-btn pl-btn-primary" onClick={onEvolve} disabled={isRunning}>
        Evolve Pipeline
      </button>
    </div>
  );
}
