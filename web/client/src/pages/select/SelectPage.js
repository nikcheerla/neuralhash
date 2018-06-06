import React from "react";
import "./select.css";
import Link from "react-router-dom/Link";
import { Button, Row, Col, Tooltip } from "antd";
import TestMorph from '../../components/TestMorph'

export default class SelectPage extends React.Component {
  state = {
    page: ""
  };
  render() {
    return (
      <div>
        <Row>
          <Col span={6}>
            {" "}
            <Link to="/auth">
              <Tooltip placement="bottom" title={"Sign Image"}>
                <Button
                  type="dashed"
                  className="bigButton"
                  shape="circle"
                  icon="lock"
                />
              </Tooltip>
            </Link>
          </Col>
          <Col span={12} />
          <Col span={6}>
            <Link to="/decode">
              <Tooltip placement="bottom" title={"Decode Image"}>
                <Button
                  type="dashed"
                  className="bigButton"
                  shape="circle"
                  icon="user"
                />
              </Tooltip>
            </Link>
          </Col>
        </Row>
      </div>
    );
  }
}
