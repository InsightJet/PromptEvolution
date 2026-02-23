import React from 'react';
import { createRoot } from 'react-dom/client';
import App from './App';
import './styles/pipeline.css';

// Mount into #pipeline-root when it exists
function mount() {
  const root = document.getElementById('pipeline-root');
  if (root && !root._reactRoot) {
    root._reactRoot = createRoot(root);
    root._reactRoot.render(<App />);
  }
}

// Auto-mount on load
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', mount);
} else {
  mount();
}

// Expose mount function for vanilla JS tab switching
window.__mountPipelineBuilder = mount;
