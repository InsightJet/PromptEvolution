import React, { useRef, useEffect } from 'react';

export default function EvolutionOverlay({ evolution, onStop }) {
  const logsEndRef = useRef(null);

  useEffect(() => {
    logsEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [evolution.logs]);

  const delta =
    evolution.initialScore != null && evolution.bestScore != null
      ? evolution.bestScore - evolution.initialScore
      : null;

  return (
    <div className="pl-overlay visible">
      <div className="pl-overlay-card">
        <h3 className="pl-overlay-title">Pipeline Evolution Running</h3>

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
            <div className="pl-score-label">Current Best</div>
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

        {/* Round Info */}
        <div className="pl-round-info">
          Round {evolution.currentRound || 0} / 3
          {evolution.currentNodeEvolving && (
            <span className="pl-evolving-label">
              Evolving: <strong>{evolution.currentNodeEvolving}</strong>
            </span>
          )}
        </div>

        {/* Progress Bar */}
        <div className="pl-progress-bar">
          <div
            className="pl-progress-fill"
            style={{ width: `${((evolution.currentRound || 0) / 3) * 100}%` }}
          />
        </div>
      </div>

      {/* Logs */}
      <div className="pl-overlay-card">
        <h4 className="pl-overlay-subtitle">Evolution Log</h4>
        <div className="pl-logs-container">
          {evolution.logs.map((entry, idx) => (
            <div key={idx} className="pl-log-entry">
              {entry.message}
            </div>
          ))}
          <div ref={logsEndRef} />
        </div>
      </div>

      <div className="pl-overlay-actions">
        <button className="pl-btn pl-btn-danger" onClick={onStop}>
          Stop Evolution
        </button>
      </div>
    </div>
  );
}
