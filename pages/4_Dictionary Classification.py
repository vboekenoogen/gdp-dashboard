import React, { useState, useCallback } from 'react';
import { Upload, FileText, Brain, Filter, Download, Plus, X, Play, Check } from 'lucide-react';

const DictionaryClassificationBot = () => {
  const [step, setStep] = useState(1);
  const [tacticDefinition, setTacticDefinition] = useState('');
  const [csvData, setCsvData] = useState([]);
  const [csvText, setCsvText] = useState('');
  const [dictionaryPrompt, setDictionaryPrompt] = useState(
    'Generate a list of single-word (unigram) keywords for a text classification dictionary focused on the "tactic" based on the "context"'
  );
  const [dictionary, setDictionary] = useState([]);
  const [isGenerating, setIsGenerating] = useState(false);
  const [classificationResults, setClassificationResults] = useState([]);
  const [newKeyword, setNewKeyword] = useState('');

  // Parse CSV data
  const parseCsv = useCallback((text) => {
    const lines = text.trim().split('\n');
    if (lines.length < 2) return [];
    
    const headers = lines[0].split(',').map(h => h.trim().replace(/"/g, ''));
    const idIndex = headers.findIndex(h => h.toLowerCase().includes('id'));
    const statementIndex = headers.findIndex(h => h.toLowerCase().includes('statement'));
    
    if (idIndex === -1 || statementIndex === -1) {
      throw new Error('CSV must contain "ID" and "Statement" columns');
    }
    
    return lines.slice(1).map((line, index) => {
      const values = line.split(',').map(v => v.trim().replace(/"/g, ''));
      return {
        id: values[idIndex] || `row_${index + 1}`,
        statement: values[statementIndex] || ''
      };
    }).filter(row => row.statement);
  }, []);

  const handleCsvUpload = (text) => {
    try {
      const parsed = parseCsv(text);
      setCsvData(parsed);
      setCsvText(text);
    } catch (error) {
      alert(`Error parsing CSV: ${error.message}`);
    }
  };

  // Generate dictionary using Claude
  const generateDictionary = async () => {
    if (!tacticDefinition) {
      alert('Please enter a tactic definition first');
      return;
    }

    setIsGenerating(true);
    try {
      const sampleStatements = csvData.slice(0, 5).map(row => row.statement).join('\n');
      const prompt = `
${dictionaryPrompt}

Tactic: "${tacticDefinition}"

Context (sample statements):
${sampleStatements}

Please respond with ONLY a JSON array of single-word keywords, like this:
["keyword1", "keyword2", "keyword3"]

Do not include any explanations or additional text. The response must be valid JSON.
      `;

      const response = await window.claude.complete(prompt);
      const keywords = JSON.parse(response.trim());
      
      if (Array.isArray(keywords)) {
        setDictionary(keywords.filter(k => k && typeof k === 'string'));
        setStep(4);
      } else {
        throw new Error('Invalid response format');
      }
    } catch (error) {
      alert(`Error generating dictionary: ${error.message}`);
    } finally {
      setIsGenerating(false);
    }
  };

  // Add keyword to dictionary
  const addKeyword = () => {
    if (newKeyword.trim() && !dictionary.includes(newKeyword.trim().toLowerCase())) {
      setDictionary([...dictionary, newKeyword.trim().toLowerCase()]);
      setNewKeyword('');
    }
  };

  // Remove keyword from dictionary
  const removeKeyword = (keyword) => {
    setDictionary(dictionary.filter(k => k !== keyword));
  };

  // Classify statements
  const classifyStatements = () => {
    const results = csvData.map(row => {
      const statement = row.statement.toLowerCase();
      const matches = dictionary.filter(keyword => 
        statement.includes(keyword.toLowerCase())
      );
      
      return {
        ...row,
        matches,
        score: matches.length,
        isMatch: matches.length > 0
      };
    });
    
    setClassificationResults(results);
    setStep(5);
  };

  // Highlight keywords in text
  const highlightKeywords = (text, keywords) => {
    if (!keywords.length) return text;
    
    let highlightedText = text;
    keywords.forEach(keyword => {
      const regex = new RegExp(`\\b${keyword}\\b`, 'gi');
      highlightedText = highlightedText.replace(regex, `<mark class="bg-yellow-200 px-1 rounded">${keyword}</mark>`);
    });
    
    return highlightedText;
  };

  // Export results
  const exportResults = () => {
    const csv = [
      'ID,Statement,Matches,Score,Classification',
      ...classificationResults.map(result => 
        `"${result.id}","${result.statement}","${result.matches.join('; ')}",${result.score},"${result.isMatch ? 'Match' : 'No Match'}"`
      )
    ].join('\n');
    
    const blob = new Blob([csv], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'classification_results.csv';
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="max-w-6xl mx-auto p-6 bg-white">
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-gray-900 mb-2">Dictionary Classification Bot</h1>
        <p className="text-gray-600">Create custom dictionaries and classify text data using AI-powered keyword generation</p>
      </div>

      {/* Progress Steps */}
      <div className="flex items-center justify-between mb-8 bg-gray-50 p-4 rounded-lg">
        {[
          { num: 1, title: 'Define Tactic', icon: FileText },
          { num: 2, title: 'Upload Data', icon: Upload },
          { num: 3, title: 'Generate Dictionary', icon: Brain },
          { num: 4, title: 'Edit Dictionary', icon: Filter },
          { num: 5, title: 'View Results', icon: Check }
        ].map(({ num, title, icon: Icon }) => (
          <div key={num} className="flex items-center">
            <div className={`w-10 h-10 rounded-full flex items-center justify-center ${
              step >= num ? 'bg-blue-500 text-white' : 'bg-gray-200 text-gray-500'
            }`}>
              {step > num ? <Check size={20} /> : <Icon size={20} />}
            </div>
            <span className={`ml-2 text-sm font-medium ${
              step >= num ? 'text-blue-600' : 'text-gray-500'
            }`}>{title}</span>
            {num < 5 && <div className="w-8 h-px bg-gray-300 mx-4" />}
          </div>
        ))}
      </div>

      {/* Step 1: Tactic Definition */}
      {step === 1 && (
        <div className="bg-white border rounded-lg p-6">
          <h2 className="text-xl font-semibold mb-4 flex items-center">
            <FileText className="mr-2" />
            Step 1: Define Your Tactic
          </h2>
          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Tactic Definition
              </label>
              <textarea
                value={tacticDefinition}
                onChange={(e) => setTacticDefinition(e.target.value)}
                placeholder="Enter a clear definition of the tactic you want to classify..."
                className="w-full p-3 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                rows={4}
              />
            </div>
            <button
              onClick={() => setStep(2)}
              disabled={!tacticDefinition.trim()}
              className="bg-blue-500 text-white px-6 py-2 rounded-md hover:bg-blue-600 disabled:bg-gray-300 disabled:cursor-not-allowed"
            >
              Next Step
            </button>
          </div>
        </div>
      )}

      {/* Step 2: Data Upload */}
      {step === 2 && (
        <div className="bg-white border rounded-lg p-6">
          <h2 className="text-xl font-semibold mb-4 flex items-center">
            <Upload className="mr-2" />
            Step 2: Upload Sample Data
          </h2>
          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                CSV Data (ID, Statement columns required)
              </label>
              <textarea
                value={csvText}
                onChange={(e) => {
                  setCsvText(e.target.value);
                  if (e.target.value.trim()) {
                    handleCsvUpload(e.target.value);
                  }
                }}
                placeholder="Paste your CSV data here or type it directly...&#10;ID,Statement&#10;1,This is a sample statement&#10;2,Another example statement"
                className="w-full p-3 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-transparent font-mono text-sm"
                rows={8}
              />
            </div>
            {csvData.length > 0 && (
              <div className="bg-green-50 p-3 rounded-md">
                <p className="text-green-800 font-medium">
                  ✓ Successfully parsed {csvData.length} statements
                </p>
                <div className="mt-2 text-sm text-green-700">
                  Preview: {csvData.slice(0, 3).map(row => `"${row.statement.substring(0, 50)}..."`).join(', ')}
                </div>
              </div>
            )}
            <div className="flex space-x-3">
              <button
                onClick={() => setStep(1)}
                className="border border-gray-300 text-gray-700 px-6 py-2 rounded-md hover:bg-gray-50"
              >
                Back
              </button>
              <button
                onClick={() => setStep(3)}
                disabled={csvData.length === 0}
                className="bg-blue-500 text-white px-6 py-2 rounded-md hover:bg-blue-600 disabled:bg-gray-300 disabled:cursor-not-allowed"
              >
                Next Step
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Step 3: Generate Dictionary */}
      {step === 3 && (
        <div className="bg-white border rounded-lg p-6">
          <h2 className="text-xl font-semibold mb-4 flex items-center">
            <Brain className="mr-2" />
            Step 3: Generate Dictionary
          </h2>
          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Dictionary Creation Prompt
              </label>
              <textarea
                value={dictionaryPrompt}
                onChange={(e) => setDictionaryPrompt(e.target.value)}
                className="w-full p-3 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                rows={3}
              />
            </div>
            <div className="bg-blue-50 p-4 rounded-md">
              <h3 className="font-medium text-blue-900 mb-2">Preview</h3>
              <p className="text-sm text-blue-800">
                <strong>Tactic:</strong> {tacticDefinition}
              </p>
              <p className="text-sm text-blue-800 mt-1">
                <strong>Sample Data:</strong> {csvData.length} statements ready for analysis
              </p>
            </div>
            <div className="flex space-x-3">
              <button
                onClick={() => setStep(2)}
                className="border border-gray-300 text-gray-700 px-6 py-2 rounded-md hover:bg-gray-50"
              >
                Back
              </button>
              <button
                onClick={generateDictionary}
                disabled={isGenerating}
                className="bg-blue-500 text-white px-6 py-2 rounded-md hover:bg-blue-600 disabled:bg-gray-300 disabled:cursor-not-allowed flex items-center"
              >
                {isGenerating ? (
                  <>
                    <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-2"></div>
                    Generating...
                  </>
                ) : (
                  <>
                    <Play className="mr-2" size={16} />
                    Generate Dictionary
                  </>
                )}
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Step 4: Edit Dictionary */}
      {step === 4 && (
        <div className="bg-white border rounded-lg p-6">
          <h2 className="text-xl font-semibold mb-4 flex items-center">
            <Filter className="mr-2" />
            Step 4: Edit Dictionary
          </h2>
          <div className="space-y-4">
            <div className="flex space-x-2">
              <input
                type="text"
                value={newKeyword}
                onChange={(e) => setNewKeyword(e.target.value)}
                onKeyPress={(e) => e.key === 'Enter' && addKeyword()}
                placeholder="Add new keyword..."
                className="flex-1 p-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              />
              <button
                onClick={addKeyword}
                className="bg-green-500 text-white px-4 py-2 rounded-md hover:bg-green-600 flex items-center"
              >
                <Plus size={16} className="mr-1" />
                Add
              </button>
            </div>
            
            <div className="bg-gray-50 p-4 rounded-md">
              <h3 className="font-medium text-gray-900 mb-3">
                Dictionary Keywords ({dictionary.length})
              </h3>
              <div className="flex flex-wrap gap-2">
                {dictionary.map((keyword, index) => (
                  <span
                    key={index}
                    className="bg-blue-100 text-blue-800 px-3 py-1 rounded-full text-sm flex items-center"
                  >
                    {keyword}
                    <button
                      onClick={() => removeKeyword(keyword)}
                      className="ml-2 text-blue-600 hover:text-blue-800"
                    >
                      <X size={14} />
                    </button>
                  </span>
                ))}
              </div>
            </div>
            
            <div className="flex space-x-3">
              <button
                onClick={() => setStep(3)}
                className="border border-gray-300 text-gray-700 px-6 py-2 rounded-md hover:bg-gray-50"
              >
                Back
              </button>
              <button
                onClick={classifyStatements}
                disabled={dictionary.length === 0}
                className="bg-blue-500 text-white px-6 py-2 rounded-md hover:bg-blue-600 disabled:bg-gray-300 disabled:cursor-not-allowed"
              >
                Classify Statements
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Step 5: Results */}
      {step === 5 && (
        <div className="bg-white border rounded-lg p-6">
          <div className="flex justify-between items-center mb-4">
            <h2 className="text-xl font-semibold flex items-center">
              <Check className="mr-2" />
              Classification Results
            </h2>
            <button
              onClick={exportResults}
              className="bg-green-500 text-white px-4 py-2 rounded-md hover:bg-green-600 flex items-center"
            >
              <Download size={16} className="mr-2" />
              Export CSV
            </button>
          </div>
          
          <div className="mb-4 grid grid-cols-3 gap-4 text-center">
            <div className="bg-blue-50 p-3 rounded-md">
              <div className="text-2xl font-bold text-blue-600">{classificationResults.length}</div>
              <div className="text-sm text-blue-800">Total Statements</div>
            </div>
            <div className="bg-green-50 p-3 rounded-md">
              <div className="text-2xl font-bold text-green-600">
                {classificationResults.filter(r => r.isMatch).length}
              </div>
              <div className="text-sm text-green-800">Matches Found</div>
            </div>
            <div className="bg-orange-50 p-3 rounded-md">
              <div className="text-2xl font-bold text-orange-600">{dictionary.length}</div>
              <div className="text-sm text-orange-800">Keywords Used</div>
            </div>
          </div>

          <div className="space-y-3 max-h-96 overflow-y-auto">
            {classificationResults.map((result, index) => (
              <div
                key={index}
                className={`p-4 rounded-md border ${
                  result.isMatch ? 'bg-green-50 border-green-200' : 'bg-gray-50 border-gray-200'
                }`}
              >
                <div className="flex justify-between items-start mb-2">
                  <span className="font-medium text-sm text-gray-600">ID: {result.id}</span>
                  <div className="flex items-center space-x-2">
                    <span className={`px-2 py-1 rounded text-xs font-medium ${
                      result.isMatch ? 'bg-green-100 text-green-800' : 'bg-gray-100 text-gray-600'
                    }`}>
                      {result.isMatch ? 'Match' : 'No Match'}
                    </span>
                    <span className="text-xs text-gray-500">Score: {result.score}</span>
                  </div>
                </div>
                <div 
                  className="text-sm text-gray-800"
                  dangerouslySetInnerHTML={{
                    __html: highlightKeywords(result.statement, result.matches)
                  }}
                />
                {result.matches.length > 0 && (
                  <div className="mt-2 text-xs text-gray-600">
                    <strong>Matched keywords:</strong> {result.matches.join(', ')}
                  </div>
                )}
              </div>
            ))}
          </div>
          
          <div className="mt-6 flex space-x-3">
            <button
              onClick={() => setStep(4)}
              className="border border-gray-300 text-gray-700 px-6 py-2 rounded-md hover:bg-gray-50"
            >
              Edit Dictionary
            </button>
            <button
              onClick={() => {
                setStep(1);
                setTacticDefinition('');
                setCsvData([]);
                setCsvText('');
                setDictionary([]);
                setClassificationResults([]);
              }}
              className="bg-blue-500 text-white px-6 py-2 rounded-md hover:bg-blue-600"
            >
              Start New Classification
            </button>
          </div>
        </div>
      )}
    </div>
  );
};

export default DictionaryClassificationBot;
