import fire from "../Config/fire";

export let apiUrl = "http://127.0.0.1:5000/";

fire
  .database()
  .ref("url")
  .on("value", (data) => {
    console.log(data.val());
    apiUrl = data.val();
  });
