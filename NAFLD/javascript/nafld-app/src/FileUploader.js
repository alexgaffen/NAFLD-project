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
  const fileInputRef = useRef(null);

  // Handle file selection
  const validImageExtensions = ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp'];
  const [chartData, setChartData] = useState([]);


  useEffect(() => {
    // Initialize Resumable.js
    resumableRef.current = new Resumable({
      target: 'http://localhost:5000/fullFileUpload',
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
        let downloadfilename = '_result.csv';
        if (disposition) {
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
        setLoading(false);
        throw new Error('Failed to fetch file');
      }
    } catch (error) {
      setLoading(false);
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

  const handleZoneClick = () => {
    fileInputRef.current.click();
  }

  return (
    <div className="uploader-container">
    
      {/* Upload Area - Hide if results are showing, typically */}
      {chartData.length === 0 && (
        <>
            <div 
                className={`drop-zone ${filesSelected ? 'active' : ''}`}
                onClick={handleZoneClick}
            >
                 <svg className="icon-upload" viewBox="0 0 24 24">
                    <path d="M19.35 10.04C18.67 6.59 15.64 4 12 4 9.11 4 6.6 5.64 5.35 8.04 2.34 8.36 0 10.91 0 14c0 3.31 2.69 6 6 6h13c2.76 0 5-2.24 5-5 0-2.64-2.05-4.78-4.65-4.96zM14 13v4h-4v-4H7l5-5 5 5h-3z"/>
                </svg>
                <h3>
                    {filesSelected ? "Files Selected" : "Click to Upload Image"}
                </h3>
                <p>Supported: PNG, JPG, JPEG, GIF, BMP, WEBP</p>
                <input 
                    type="file" 
                    onChange={handleFileSelect} 
                    multiple 
                    className="file-input"
                    ref={fileInputRef}
                />
            </div>

            {/* Preview Section */}
            {image && (
                <div className="preview-container">
                    <p className="mb-1 font-bold">Preview:</p>
                    <img
                        alt="Selected"
                        src={URL.createObjectURL(image)}
                        className="preview-image"
                    />
                </div>
            )}

            {!image && triedSelecting && (
               <div className="preview-container">
                  <p className="text-muted">Invalid image format or multiple files.</p>
                  <img
                    alt="Info"
                    src="/Images/combined.png"
                    className="preview-image"
                    style={{maxWidth: '100%', height:'auto'}}
                  />
               </div>
            )}

             {/* Action Buttons */}
            {filesSelected && (
                <div className="mt-2">
                     <button 
                        onClick={handleSubmit} 
                        className="btn btn-primary"
                        disabled={loading || (progress > 0 && progress < 100)}
                     >
                        {loading ? 'Processing...' : 'Upload & Analyze Files'}
                     </button>
                </div>
            )}

            {/* Progress Bar */}
            {progress && parseInt(progress) < 100 && (
                 <div className="mt-2">
                     <p>Uploading: {parseInt(progress)}%</p>
                     <ProgressBar progress={progress} />
                 </div>
            )}
        </>
      )}

      {/* Post-Upload / Processing Section */}
      {fileUploaded && chartData.length === 0 && (
        <div className="card mt-2">
           <h3>File Uploaded Successfully!</h3>
           <p className="text-muted mb-2">Filename: {fileName}</p>
           
           <button 
                className="btn btn-primary" 
                onClick={() => downloadFile(fileName)}
                disabled={loading}
            >
               {loading ? 'Processing Analysis...' : 'Process Analysis & View Results'}
           </button>
        </div>
      )}
      
      {loading && chartData.length === 0 && !filesSelected && (
         <div className="mt-2">
             <p>Analyzing on server...</p>
         </div>
      )}


      {/* Charts / Results Section */}
      {chartData.length > 0 && (
          <div className="charts-grid">
            {chartData.map((row, index) => (
            <div key={index} className="chart-card">
                <h3>{row.image_name}</h3> 
                <ResponsiveContainer width="100%" height={300}>
                <BarChart data={[
                    { name: "None", value: row.None },
                    { name: "Perisinu", value: row.Perisinusoidal },
                    { name: "Bridging", value: row.Bridging },
                    { name: "Cirrosis", value: row.Cirrosis }
                ]}>
                    <CartesianGrid strokeDasharray="3 3" vertical={false} />
                    <XAxis dataKey="name" tick={{fontSize: 12}} />
                    <YAxis 
                        domain={[0, 1]} 
                        tickFormatter={(tick) => `${(tick * 100).toFixed(0)}%`} 
                        tick={{fontSize: 12}}
                    />
                    <Tooltip formatter={(value) => `${(value * 100).toFixed(1)}%`} cursor={{fill: '#f0f0f0'}} />
                    <Bar dataKey="value" fill="#7A003C" radius={[4, 4, 0, 0]} />
                </BarChart>
                </ResponsiveContainer>
            </div>
            ))}
            
            <div className="col-span-full text-center mt-2">
                <button 
                    className="btn btn-primary" 
                    onClick={() => window.location.reload()}
                >
                    Analyze Another Image
                </button>
            </div>
          </div>
      )}

    </div>
  );
}

export default FileUploader;

