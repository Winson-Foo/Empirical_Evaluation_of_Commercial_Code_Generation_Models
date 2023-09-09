// Route config

import React from 'react'
import { Switch, Route } from 'react-router-dom'

// route components

import Home from '../components/Home'
import About from '../components/About'
import Contact from '../components/Contact'

function RouteConfig() {
  return (
    <Switch>
      <Route path="/about">
        <About />
      </Route>
      <Route path="/contact">
        <Contact />
      </Route>
      <Route path="/">
        <Home />
      </Route>
    </Switch>
  )
}

export default RouteConfig