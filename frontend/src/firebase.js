import { initializeApp } from "firebase/app";
import { getFirestore } from "firebase/firestore";

// https://firebase.google.com/docs/web/setup#available-libraries

// Your web app's Firebase configuration
// For Firebase JS SDK v7.20.0 and later, measurementId is optional
const firebaseConfig = {
    apiKey: "AIzaSyA2E87BkuFhlnDwszfcq-SfUgWZ-A5Bc6w",
    authDomain: "proyecto-ia3-ff33c.firebaseapp.com",
    projectId: "proyecto-ia3-ff33c",
    storageBucket: "proyecto-ia3-ff33c.firebasestorage.app",
    messagingSenderId: "729548029871",
    appId: "1:729548029871:web:3c352a0796a8d87900f95a",
    measurementId: "G-7H8PXYC93P"
};

const app = initializeApp(firebaseConfig);
export const db = getFirestore(app);

