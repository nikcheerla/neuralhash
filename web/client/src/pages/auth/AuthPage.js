import React from "react";
import FacebookAuth from 'react-facebook-auth';
import Cookies from 'js-cookie';
import {Button} from 'antd';
export default class AuthPage extends React.Component {
  Button = ({ onClick }) => (
    <Button type="primary" onClick={onClick} icon="facebook">Login With Facebook</Button>
  );

  authenticate = response => {
    if(response.userID) {
      Cookies.set('facebook-id', response.userID);
      this.props.history.push("/sign");
    }
  };

  render() {
    return (
      <div>
        <h1>Facebook Auth</h1>
        <FacebookAuth
          appId={1951426491774080}
          autoLoad={true}
          callback={this.authenticate}
          component={this.Button}
        />
      </div>
    );
  }
}
