import React, { useState } from 'react';
import { Upload } from 'lucide-react';

export default function SpliceSampleFinder() {
  const [isUploading, setIsUploading] = useState(false);
  const [uploadedSample, setUploadedSample] = useState(null);
  const [matches, setMatches] = useState(null);
  const [error, setError] = useState(null);
  
  // File selection handler
  const handleFileChange = (e) => {
    if (e.target.files.length > 0) {
      const file = e.target.files[0];
      setUploadedSample({
        file: file,
        name: file.name,
        url: URL.createObjectURL(file)
      });
    }
  };

  // Form submission handler
  const handleSubmit = async () => {
    if (!uploadedSample) return;

    setIsUploading(true);
    setError(null);
    
    try {
      const formData = new FormData();
      formData.append('sample', uploadedSample.file);
      
      const response = await fetch('/api/find-similar', {
        method: 'POST',
        body: formData,
      });
      
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || 'Failed to process the audio sample');
      }
      
      const data = await response.json();
      
      setUploadedSample({
        ...uploadedSample,
        url: data.uploadedUrl
      });
      
      setMatches(data.matches);
    } catch (err) {
      setError(err.message || 'An error occurred while processing your request');
      console.error(err);
    } finally {
      setIsUploading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 to-gray-800 text-white p-6">
      <div className="max-w-3xl mx-auto">
        <header className="mb-12 text-center">
          <h1 className="text-4xl font-bold mb-2 bg-clip-text text-transparent bg-gradient-to-r from-cyan-400 to-purple-500">
            SherlockBeat
          </h1>
          <p className="text-gray-400">
            Upload an audio sample to find similar sounds in our library
          </p>
        </header>

        <div className="mb-10 p-6 bg-gray-800/50 rounded-xl backdrop-blur-sm border border-gray-700">
          <div className="flex flex-col items-center justify-center">
            <label 
              htmlFor="sample-upload" 
              className="w-full h-32 border-2 border-dashed border-gray-600 rounded-lg flex flex-col items-center justify-center cursor-pointer hover:border-cyan-400 transition-colors group"
            >
              <Upload className="w-10 h-10 mb-2 text-gray-500 group-hover:text-cyan-400 transition-colors" />
              <span className="text-gray-400 group-hover:text-white transition-colors">
                {uploadedSample ? uploadedSample.name : "Choose an audio file or drag & drop"}
              </span>
              <input 
                type="file" 
                id="sample-upload" 
                name="sample" 
                accept="audio/*" 
                className="hidden" 
                onChange={handleFileChange}
              />
            </label>
            
            <button 
              onClick={handleSubmit} 
              disabled={isUploading || !uploadedSample}
              className={`mt-6 px-6 py-3 rounded-lg font-medium flex items-center ${
                isUploading || !uploadedSample 
                  ? 'bg-gray-700 text-gray-400 cursor-not-allowed' 
                  : 'bg-gradient-to-r from-cyan-500 to-purple-600 text-white hover:shadow-lg hover:shadow-cyan-500/20 transition-all'
              }`}
            >
              {isUploading ? (
                <>
                  <svg className="animate-spin -ml-1 mr-3 h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                  </svg>
                  Processing...
                </>
              ) : (
                <>
                  <svg className="mr-2 w-5 h-5" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 01-3-3V5a3 3 0 116 0v6a3 3 0 01-3 3z" />
                  </svg>
                  Find Similar Samples
                </>
              )}
            </button>
          </div>
        </div>

        {uploadedSample && (
          <div className="mb-10 p-6 bg-gray-800/50 rounded-xl backdrop-blur-sm border border-gray-700">
            <h2 className="text-xl font-semibold mb-4 flex items-center">
              <svg className="mr-2 w-5 h-5 text-cyan-400" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 01-3-3V5a3 3 0 116 0v6a3 3 0 01-3 3z" />
              </svg>
              Uploaded Sample
            </h2>
            <div className="p-4 bg-gray-800 rounded-lg">
              <p className="mb-2 text-gray-400">{uploadedSample.name}</p>
              <audio 
                controls 
                className="w-full h-12"
                src={uploadedSample.url}
              />
            </div>
          </div>
        )}

        {matches && (
          <div className="p-6 bg-gray-800/50 rounded-xl backdrop-blur-sm border border-gray-700">
            <h2 className="text-xl font-semibold mb-4">Top Matches</h2>
            
            {matches.length > 0 ? (
              <div className="space-y-4">
                {matches.map((match, index) => (
                  <div 
                    key={index} 
                    className="p-4 bg-gray-800 rounded-lg hover:bg-gray-700 transition-colors"
                  >
                    <div className="flex flex-wrap items-center justify-between mb-2">
                      <span className="font-medium">{match.filename}</span>
                      <span className="px-3 py-1 bg-gray-700 rounded-full text-sm">
                        Similarity: <span className="text-cyan-400 font-medium">{match.score.toFixed(3)}</span>
                      </span>
                    </div>
                    <audio 
                      controls 
                      className="w-full h-12"
                      src={match.url}
                    />
                  </div>
                ))}
              </div>
            ) : (
              <p className="text-center text-gray-400 py-6">No similar samples found.</p>
            )}
          </div>
        )}
        
        {error && (
          <div className="p-6 mb-6 bg-red-900/30 rounded-xl backdrop-blur-sm border border-red-800">
            <p className="text-red-300">{error}</p>
          </div>
        )}
      </div>
    </div>
  );
}