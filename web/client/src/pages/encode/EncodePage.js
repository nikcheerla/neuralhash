import React from "react";
import UploadFileComponent from "../../components/UploadFile";
import Cookies from "js-cookie";

export default class EncodePage extends React.Component {
  render() {
    return (
      <div>
        <h1>Sign File</h1>
        <UploadFileComponent
          endpoint="encode"
          data={Cookies.get("facebook-id")}
          onResponse={this.downloadFile}
        />
      </div>
    );
  }

  downloadFile = (file, fileName) => {
    console.log("Starting timeout...");
    console.log(file);
    setTimeout(async () => {
      let a = document.createElement("a");
      document.body.appendChild(a);
      a.style = "display: none";
      let url = window.URL.createObjectURL(await file.blob());
      a.href = url;
      a.download = "protected_" + fileName;
      a.click();
      window.URL.revokeObjectURL(url);
    }, 2000);
  };

  handleFormChange(event) {
    this.setState({ value: event.target.value });
  }
}
