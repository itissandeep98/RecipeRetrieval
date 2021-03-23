import firebase from "firebase/app";
import "firebase/firestore";
require("firebase/auth");
require("firebase/database");
require("firebase/storage");

var firebaseConfig = {
  apiKey: "AIzaSyAAYmzTCsgF4r9wXv0Tv0GwT3j0mqsM_eQ",
  authDomain: "ir-project-d1527.firebaseapp.com",
  projectId: "ir-project-d1527",
  storageBucket: "ir-project-d1527.appspot.com",
  messagingSenderId: "437213427392",
  appId: "1:437213427392:web:036fad3d28f61d4a481f25",
};

const fire = firebase.initializeApp(firebaseConfig);

export const storage = firebase.storage();
export default fire;
