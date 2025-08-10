import React from 'react';
import ReactDOM from 'react-dom/client';
import './index.css';
// import AppLocal from './AppLocal'; // Original complex version
import AppAC from './AppAC'; // Simplified AC version

const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(
  <React.StrictMode>
    <AppAC />
  </React.StrictMode>
);