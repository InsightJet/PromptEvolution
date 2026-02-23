import React, { useState } from 'react';

export default function ResultsOverlay({ evolution, pipeline, onClose, onApply }) {
  const [showDiffs, setShowDiffs] = useState(false);

  const delta =
    evolution.initialScore != null && evolution.bestScore != null
      ? evolution.bestScore - evolution.initialScore
      : null;

  const evolvedNodes = evolution.evolvedPipeline?.nodes || [];

  return (
    <div className="pl-overlay visible">
      <div className="pl-overlay-card">
        <h3 className="pl-overlay-title">
          {evolution.status === 'completed' ? 'Evolution Complete' : 'Evolution Stopped'}
        </h3>

        {/* Score Banner */}
        <div className="pl-score-banner">
          <div className="pl-score-item">
            <div className="pl-score-label">Original</div>
            <div className="pl-score-value">
              {evolution.initialScore != null ? evolution.initialScore.toFixed(1) : '\u2014'}
            </div>
          </div>
          <div className="pl-score-arrow">&rarr;</div>
          <div className="pl-score-item">
            <div className="pl-score-label">Best</div>
            <div className="pl-score-value best">
              {evolution.bestScore != null ? evolution.bestScore.toFixed(1) : '\u2014'}
            </div>
          </div>
          {delta != null && delta !== 0 && (
            <div className={`pl-score-delta ${delta > 0 ? 'positive' : 'negative'}`}>
              {delta > 0 ? '+' : ''}
              {delta.toFixed(1)}
            </div>
          )}
        </div>
      </div>

      {/* Node Evolution Log */}
      {evolution.nodeEvolutionLog && evolution.nodeEvolutionLog.length > 0 && (
        <div className="pl-overlay-card">
          <h4 className="pl-overlay-subtitle">Node Evolution Summary</h4>
          <div className="pl-node-log">
            {evolution.nodeEvolutionLog.map((entry, idx) => (
              <div key={idx} className="pl-node-log-entry">
                <div className="pl-node-log-header">
                  <span className="pl-node-log-round">Round {entry.round}</span>
                  <span className="pl-node-log-name">{entry.node_id}</span>
                  {entry.reason && (
                    <span className="pl-node-log-reason">{entry.reason}</span>
                  )}
                </div>
                {entry.score_before != null && entry.score_after != null && (
                  <div className="pl-node-log-scores">
                    <span>{entry.score_before.toFixed(1)}</span>
                    <span className="pl-score-arrow-sm">&rarr;</span>
                    <span className={entry.score_after > entry.score_before ? 'pl-text-positive' : ''}>
                      {entry.score_after.toFixed(1)}
                    </span>
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Prompt Diffs */}
      {evolvedNodes.length > 0 && (
        <div className="pl-overlay-card">
          <div
            className="pl-diff-toggle"
            onClick={() => setShowDiffs(!showDiffs)}
          >
            <h4 className="pl-overlay-subtitle">Evolved Prompts</h4>
            <span className="pl-collapse-icon">{showDiffs ? '\u25BC' : '\u25B6'}</span>
          </div>

          {showDiffs && (
            <div className="pl-diff-list">
              {evolvedNodes.map((eNode) => {
                const original = pipeline.nodes.find((n) => n.id === eNode.id);
                const originalPrompt = original?.data?.promptTemplate || '';
                const evolvedPrompt = eNode.prompt_template || '';
                const changed = originalPrompt !== evolvedPrompt;

                return (
                  <div key={eNode.id} className="pl-diff-entry">
                    <div className="pl-diff-header">
                      <span className="pl-diff-name">{eNode.label || eNode.id}</span>
                      {changed ? (
                        <span className="pl-diff-badge changed">Changed</span>
                      ) : (
                        <span className="pl-diff-badge unchanged">Unchanged</span>
                      )}
                    </div>
                    {changed && (
                      <div className="pl-diff-content">
                        <div className="pl-diff-block original">
                          <label className="pl-label-sm">Original</label>
                          <pre className="pl-pre-sm">{originalPrompt}</pre>
                        </div>
                        <div className="pl-diff-block evolved">
                          <label className="pl-label-sm">Evolved</label>
                          <pre className="pl-pre-sm">{evolvedPrompt}</pre>
                        </div>
                      </div>
                    )}
                  </div>
                );
              })}
            </div>
          )}
        </div>
      )}

      {/* Actions */}
      <div className="pl-overlay-actions">
        <button className="pl-btn pl-btn-secondary" onClick={onClose}>
          Dismiss
        </button>
        {evolvedNodes.length > 0 && (
          <button className="pl-btn pl-btn-primary" onClick={onApply}>
            Apply Evolved Prompts
          </button>
        )}
      </div>
    </div>
  );
}
