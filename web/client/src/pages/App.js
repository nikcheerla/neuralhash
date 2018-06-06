import React, { Component } from "react";
import AuthPage from "./auth/AuthPage";
import SelectPage from "./select/SelectPage";
import EncodePage from "./encode/EncodePage";
import DecodePage from "./decode/DecodePage";
import "./global.css";
import Router from "react-router-dom/BrowserRouter";
import { AnimatedSwitch, spring } from "react-router-transition";
import Route from "react-router-dom/Route";
import ImageTrace from '../components/ImageTrace/ImageTrace'


function mapStyles(styles) {
  return {
    opacity: styles.opacity,
    transform: `scale(${styles.scale})`
  };
}

// wrap the `spring` helper to use a bouncy config
function bounce(val) {
  return spring(val, {
    stiffness: 330,
    damping: 22
  });
}

class App extends Component {
  state = {
    page: "select"
  };

  render() {
    return (
      <div className="wrapper">
        <Router>
          <AnimatedSwitch
            atEnter={{
              opacity: 0,
              scale: 1.2
            }}
            atLeave={{
              opacity: bounce(0),
              scale: bounce(0.8)
            }}
            atActive={{
              opacity: bounce(1),
              scale: bounce(1)
            }}
            mapStyles={mapStyles}
            className="switch-wrapper"
          >
            <Route exact path="/" component={SelectPage} />
            <Route exact path="/trace" component={ImageTrace} />
            <Route path="/auth/" component={AuthPage} />
            <Route path="/sign/" component={EncodePage} />
            <Route path="/decode/" component={DecodePage} />
          </AnimatedSwitch>
        </Router>
      </div>
    );
  }
}
 
export default App;
