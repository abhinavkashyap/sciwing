import React from 'react';
import './App.css';
import { Route, BrowserRouter as Router } from 'react-router-dom'
import Home from './components/home/home';
import DataVis from './components/data_vis/data_vis';

function App() {
  return (
    <div className="App">
      <Router>
        <div>
          <Route exact path="/" component={Home} />
          <Route path='/datavis' component={DataVis}></Route>
        </div>
      </Router>
    </div>
  );
}

export default App;
