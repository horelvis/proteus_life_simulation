import React from 'react';
import ReactDOM from 'react-dom/client';
import './index.css';
import App from './App'; // Main app with mode selector

const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);