import React from "react";
import { Upload, Icon } from "antd";

export default class UploadFileComponent extends React.Component {
  render() {
    return (
      <div>
        <Upload.Dragger
          name="file"
          id="file"
          multiple={false}
          style={{ padding: "20px" }}
          beforeUpload={(info) => {
            this.submit(info, this.props.data)
            return false;
          }}
        >
          <p className="ant-upload-drag-icon">
            <Icon type="inbox" />
          </p>
          <p className="ant-upload-text">Click or drag file a to upload</p>
        </Upload.Dragger>
      </div>
    );
  }

  submit = (file, data) => {
    let fileName = document.getElementById("file").files[0].name;
    let data = await fetch("http://localhost:5000/" + this.props.endpoint, {
      method: "POST",
      body: formData
    });
    this.props.onResponse(data, fileName);
  };
}
