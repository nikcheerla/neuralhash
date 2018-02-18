import React from "react";
import UploadFileComponent from "../../components/UploadFile";

export default class DecodePage extends React.Component {
  render() {
    return (
      <div>
        <h1>Decode File</h1>
        <UploadFileComponent
          endpoint="decode"
          onResponse={async (file) => {
            alert(await file.text())
          }}
        />
      </div>
    );
  }
}
