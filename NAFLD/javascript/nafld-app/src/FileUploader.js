import React, { useEffect, useRef, useState } from 'react';
import Resumable from 'resumablejs';
import ProgressBar from './progressBar';
import Papa from 'papaparse';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';

const FileUploader = () => {
  const resumableRef = useRef(null);
  const [filesSelected, setFilesSelected] = useState(false);
  const [fileUploaded, setFileUploaded] = useState(false);
  const [fileName, setFileName] = useState("");
  const [image, setSelectedImage] = useState(null);
  const [progress, setProgress] = useState(null)
  const [loading, setLoading] = useState(false)
  const [triedSelecting, setTriedSelecting] = useState(false)

  // Handle file selection
  const validImageExtensions = ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp'];
  const [chartData, setChartData] = useState([]);


  useEffect(() => {
    // Initialize Resumable.js
    resumableRef.current = new Resumable({

      target: 'http://localhost:5000/fullFileUpload',
      // target: 'http://localhost:5000/largefile', 
      chunkSize: 3 * 1024 * 1024, // 1 MB per chunk
      simultaneousUploads: 1,
      testChunks: false, // Disable testing chunks before upload
      throttleProgressCallbacks: 1, // Progress updates
    });

    // Resumable.js event listeners
    resumableRef.current.on('fileAdded', (file) => {
      console.log("File added:", file);
      setFilesSelected(true);
    });

    resumableRef.current.on('fileSuccess', (file, message) => {
      setFileUploaded(true);
      let message_json = JSON.parse(message)
      console.log(message_json.fileName);
      setFileName(message_json.fileName)
      console.log("File upload successful:", file);
    });

    resumableRef.current.on('fileError', (file, error) => {
      console.error("File upload error:", error);
    });

    resumableRef.current.on('progress', () => {
      setProgress((resumableRef.current.progress() * 100).toFixed(2));
      console.log(`Progress: ${(resumableRef.current.progress() * 100).toFixed(2)}%`);
    });
  }, []);


  const handleFileSelect = (event) => {
    setTriedSelecting(true);
    const files = event.target.files;
    const selectedFile = files[0];

    if (selectedFile) {
      // Get the file extension (in lowercase for case-insensitivity)
      const fileExtension = selectedFile.name.split('.').pop().toLowerCase();

      // Check if the file extension is in the list of valid image extensions
      if (validImageExtensions.includes(`.${fileExtension}`)) {
        setSelectedImage(selectedFile);
      } else {
        setSelectedImage(null); 
      }
    }

    // Add files to Resumable.js
    resumableRef.current.addFiles(files);
  };

 
  const handleSubmit = (event) => {
    event.preventDefault(); 
    setFileUploaded(false);
    if (filesSelected) {
      resumableRef.current.upload();
    }
  };

  const downloadFile = async (filename) => {
    try {
      // Make a GET request to the backend with fetch
      setLoading(true);
      const response = await fetch(`http://localhost:5000/download/${filename}`, {
        method: 'GET',
      });

      // Check if the response is successful
      if (response.ok) {
        setLoading(false);

        const blob = await response.blob();

        const disposition = response.headers.get('Content-Disposition');
        console.log(response.headers)
        let downloadfilename = '_result.csv';
        if (disposition) {
          console.log(disposition);
          downloadfilename = disposition
            .split('filename=')[1]
            .replace(/"/g, '');
        }
        // Create a temporary link to trigger the download
        const url = window.URL.createObjectURL(blob);
        const link = document.createElement('a');
        link.href = url;
        link.setAttribute('download', downloadfilename);

        // Append the link to the body
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);

        const reader = new FileReader();
        reader.onload = () => {
          const text = reader.result;
          parseCSV(text); // Pass the file content to the CSV parser
        };

        // Trigger reading of the blob as text
        reader.readAsText(blob);

      } else {
        throw new Error('Failed to fetch file');
      }
    } catch (error) {
      console.error('Error downloading the file:', error);
      alert('Failed to download the file');
    }
  };

  //display results form backend
  const parseCSV = (csvText) => {
    Papa.parse(csvText, {
      header: true, 
      skipEmptyLines: true,
      complete: (result) => {
        
        const data = result.data.map(row => ({
          image_name: row.image_name,  
          None: JSON.parse(row.None),  
          Perisinusoidal: JSON.parse(row.Perisinusoidal),
          Bridging: JSON.parse(row.Bridging),
          Cirrosis: JSON.parse(row.Cirrosis),
        }));

        setChartData(data);
      },
    });
  };

  return (

    <div className="ImageSubmission">

    {/*  requires some redundancy removal, can do this in one go */}
    {image && (
        <div className="ImageSelection">
          <img
            alt="Selected"
            src={URL.createObjectURL(image)}
            style={{
              display: 'block',
              margin: '0 auto',
              width: '500px',
              height: 'auto',
            }}
          />
        </div>
      )}

      {!image && triedSelecting &&(
        <div className="ImageSelection">
          <img
            alt="Selected"
            src={"/Images/combined.png" }
            style={{
              display: 'block',
              margin: '0 auto',
              width: '1080px',
              height: 'auto',
            }}
          />
        </div>
      )}

      {fileUploaded && (
        <div className="csv-container">
          <div className="csvDownload">
            Download Result File for: {fileName}
            <button onClick={() => downloadFile(fileName)}>Download & Process</button>
            {loading && <div> Processing Input... </div>}
          </div>
        </div>
      )}


      <div className="bar-graph-container">

        {chartData.map((row, index) => (
          <div key={index} className="bar-graph">
            <h2>{row.image_name}</h2> 
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={[
                { name: "None", value: row.None },
                { name: "Periportal", value: row.Perisinusoidal },
                { name: "Bridging", value: row.Bridging },
                { name: "Cirrosis", value: row.Cirrosis }
              ]}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="name" label={{ value: "Categories", position: "bottom" }} />
                  <YAxis 
                    domain={[0, 1]} // Set the Y-axis to range from 0 to 1
                    label={{ value: "Confidence %", angle: -90, position: "insideLeft" }} 
                    tickFormatter={(tick) => `${(tick * 100).toFixed(0)}%`} 
                  />
                  <Tooltip formatter={(value) => `${(value * 100).toFixed(0)}%`} />
                <Bar dataKey="value" fill="#7A003C" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        ))}
      </div>


      <div className="form-container">
        <form onSubmit={handleSubmit}>
          <input type="file" onChange={handleFileSelect} multiple />
          <button type="submit" disabled={!filesSelected}>
            Upload Files
          </button>
        </form>
      </div>
      {progress &&
        <div className='progress-bar-container'>
          <ProgressBar progress={progress}> </ProgressBar>
        </div>
      }
    </div>


  );
}

export default FileUploader;

