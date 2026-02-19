import ImageSubmission from "./ImageSubmission";

function App() {
  return (
    <div className="App">
      <header className="centered-header">
        <img src="/Images/McMaster.png" alt="Logo Left" className="logo" />
        <img src="/Images/ICELAB.png" alt="Logo middle" className="logo" />
        <img src="/Images/Heersink.png" alt="Logo Right" className="logo" />
      </header>
        <h1 className="centered-header">
          FibroAi
        </h1>
      <div>
        <ImageSubmission />
      </div>
    </div>
  );
}

export default App;
