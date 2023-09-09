// Routes.js

import React from 'react'
import { Switch, Route } from 'react-router-dom'
import HomePage from 'views/HomePage'
import AboutPage from 'views/AboutPage'
import ContactPage from 'views/ContactPage'

const Routes = () => {
  return (
    <Switch>
      <Route exact path="/" component={HomePage} />
      <Route exact path="/about" component={AboutPage} />
      <Route exact path="/contact" component={ContactPage} />
    </Switch>
  )
}

export default Routes