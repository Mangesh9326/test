import React, { useState, useEffect, useRef } from 'react';
import { useLocation } from 'react-router-dom';
import { useCitizenStore } from '../../store/useCitizenStore';
import { Send, Bot, User, Trash2, Sparkles, Info, AlertTriangle } from 'lucide-react';

export default function CitizenAssistant() {
  const location = useLocation();
  const { askAI, isSimulated } = useCitizenStore();
  
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  
  const messagesEndRef = useRef(null);
  const inputRef = useRef(null);

  const SUGGESTED_QUESTIONS = [
    "What's happening around me?",
    "Are there any traffic problems nearby?",
    "Find hospitals near me",
    "Explain today's major incidents",
    "Find fire stations near me",
    "What city services are nearby?"
  ];

  // Handle incoming contextual questions (e.g., from Alerts or Around Me pages)
  useEffect(() => {
    if (location.state?.initialMessage && messages.length === 0) {
      handleSend(location.state.initialMessage, location.state.contextType);
      // Clean up the router state so it doesn't re-trigger on refresh
      window.history.replaceState({}, document.title);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [location.state]);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isTyping]);

  const handleSend = async (textOverride, contextType = 'general') => {
    const query = (textOverride || input).trim();
    if (!query || isTyping) return;

    const userMessage = {
      id: crypto.randomUUID(),
      role: 'user',
      content: query,
      timestamp: new Date().toISOString(),
      contextType
    };

    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setIsTyping(true);

    try {
      const reply = await askAI(query);
      const aiMessage = {
        id: crypto.randomUUID(),
        role: 'assistant',
        content: reply,
        timestamp: new Date().toISOString(),
        contextType
      };
      setMessages(prev => [...prev, aiMessage]);
    } catch (error) {
      const errorMessage = {
        id: crypto.randomUUID(),
        role: 'assistant',
        content: "I don't have confirmed information about that in the available Smart City data, or I am temporarily unavailable. Please try again later.",
        timestamp: new Date().toISOString(),
        contextType: 'error'
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsTyping(false);
      // Return focus to input on desktop, but avoid mobile keyboard jump if triggered by suggestion chip
      if (!textOverride && window.innerWidth > 768) {
        inputRef.current?.focus();
      }
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const clearChat = () => {
    if (window.confirm("Are you sure you want to clear this conversation?")) {
      setMessages([]);
    }
  };

  // Helper to safely format basic markdown-style bolding returned by Groq
  const formatContent = (text) => {
    return text.split('\n').map((line, i) => (
      <React.Fragment key={i}>
        {line.split(/(\*\*.*?\*\*)/g).map((part, j) => {
          if (part.startsWith('**') && part.endsWith('**')) {
            return <strong key={j} className="text-white font-bold">{part.slice(2, -2)}</strong>;
          }
          return part;
        })}
        {i !== text.split('\n').length - 1 && <br />}
      </React.Fragment>
    ));
  };

  return (
    <div className="flex flex-col h-full bg-slate-950 relative overflow-hidden pt-4 pb-24 lg:pb-6">
      
      {/* HEADER */}
      <div className="flex items-center justify-between px-4 pb-3 border-b border-slate-800 shrink-0">
        <div>
          <h1 className="text-2xl font-black text-white flex items-center gap-2">
            City AI Assistant <Sparkles className="w-5 h-5 text-blue-400" />
          </h1>
          <p className="text-xs text-slate-400 mt-1">Ask about your city, nearby services, traffic, incidents and alerts.</p>
          <div className="flex items-center gap-2 mt-2">
            <span className="bg-emerald-500/20 text-emerald-400 border border-emerald-500/30 px-2 py-0.5 rounded text-[10px] font-bold uppercase flex items-center gap-1">
              <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse"></span> Available
            </span>
            {isSimulated && (
              <span className="bg-blue-500/10 border border-blue-500/30 text-blue-400 text-[10px] font-bold px-2 py-0.5 rounded uppercase tracking-wider">
                Simulated Data
              </span>
            )}
          </div>
        </div>
        
        {messages.length > 0 && (
          <button 
            onClick={clearChat}
            aria-label="Clear chat"
            className="p-2 text-slate-400 hover:text-red-400 bg-slate-900 hover:bg-slate-800 rounded-xl transition-colors min-h-[44px] min-w-[44px] flex items-center justify-center"
          >
            <Trash2 className="w-5 h-5" />
          </button>
        )}
      </div>

      {/* CHAT AREA */}
      <div className="flex-1 overflow-y-auto px-4 pt-4 pb-12 space-y-6 scrollbar-hide">
        
        {/* Welcome / Empty State */}
        {messages.length === 0 && (
          <div className="flex flex-col items-center justify-center h-full text-center px-4 fade-in">
            <div className="w-16 h-16 bg-blue-500/10 border border-blue-500/20 rounded-full flex items-center justify-center mb-4">
              <Bot className="w-8 h-8 text-blue-400" />
            </div>
            <h2 className="text-xl font-bold text-white mb-2">How can I help you today?</h2>
            <p className="text-sm text-slate-400 mb-8 max-w-xs">
              I can explain city alerts, nearby services, traffic conditions and other information available in the Smart City system.
            </p>
            
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 w-full max-w-lg">
              {SUGGESTED_QUESTIONS.map((q, i) => (
                <button
                  key={i}
                  onClick={() => handleSend(q)}
                  className="bg-slate-900 border border-slate-800 hover:border-blue-500/50 text-slate-300 text-sm text-left p-4 rounded-2xl transition-colors min-h-[44px]"
                >
                  {q}
                </button>
              ))}
            </div>
          </div>
        )}

        {/* Message List */}
        {messages.map((msg) => (
          <div key={msg.id} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
            <div className={`flex gap-3 max-w-[85%] ${msg.role === 'user' ? 'flex-row-reverse' : 'flex-row'}`}>
              
              <div className="shrink-0 mt-1">
                {msg.role === 'user' ? (
                  <div className="w-8 h-8 bg-blue-600 rounded-full flex items-center justify-center">
                    <User className="w-4 h-4 text-white" />
                  </div>
                ) : (
                  <div className="w-8 h-8 bg-slate-800 border border-slate-700 rounded-full flex items-center justify-center">
                    <Bot className="w-4 h-4 text-blue-400" />
                  </div>
                )}
              </div>

              <div className="flex flex-col gap-1">
                <div 
                  className={`p-4 rounded-2xl text-sm leading-relaxed ${
                    msg.role === 'user' 
                      ? 'bg-blue-600 text-white rounded-tr-none' 
                      : 'bg-slate-900 border border-slate-800 text-slate-300 rounded-tl-none'
                  }`}
                >
                  {msg.contextType === 'alert' && msg.role === 'user' && (
                    <div className="flex items-center gap-1.5 text-[10px] uppercase font-bold tracking-wider text-blue-200 mb-2 bg-black/20 px-2 py-1 rounded inline-block">
                      <Info className="w-3 h-3 inline" /> Context Attached
                    </div>
                  )}
                  {formatContent(msg.content)}
                </div>
                {msg.role === 'assistant' && msg.contextType !== 'error' && (
                   <span className="text-[10px] text-slate-500 ml-1">AI-generated based on Smart City platform data</span>
                )}
              </div>

            </div>
          </div>
        ))}

        {/* Typing Indicator */}
        {isTyping && (
          <div className="flex justify-start">
            <div className="flex gap-3 max-w-[85%]">
              <div className="w-8 h-8 bg-slate-800 border border-slate-700 rounded-full flex items-center justify-center shrink-0 mt-1">
                <Bot className="w-4 h-4 text-blue-400" />
              </div>
              <div className="bg-slate-900 border border-slate-800 p-4 rounded-2xl rounded-tl-none flex items-center gap-2">
                <span className="text-sm text-slate-400">AI is thinking</span>
                <div className="flex gap-1 mt-1">
                  <span className="w-1.5 h-1.5 bg-blue-500 rounded-full animate-bounce"></span>
                  <span className="w-1.5 h-1.5 bg-blue-500 rounded-full animate-bounce delay-75"></span>
                  <span className="w-1.5 h-1.5 bg-blue-500 rounded-full animate-bounce delay-150"></span>
                </div>
              </div>
            </div>
          </div>
        )}
        
        <div ref={messagesEndRef} className="h-4" />
      </div>

      {/* COMPOSER */}
      <div className="absolute bottom-16 lg:bottom-0 left-0 right-0 p-4 bg-slate-950/90 backdrop-blur-md border-t border-slate-900 shrink-0" style={{ paddingBottom: 'calc(1rem + env(safe-area-inset-bottom))' }}>
        
        {/* Safety Warning */}
        <div className="flex items-center justify-center gap-1.5 mb-2 text-[10px] font-bold uppercase tracking-wider text-slate-500">
          <AlertTriangle className="w-3 h-3 text-amber-500" /> For informational purposes. Not for emergency dispatch.
        </div>

        <div className="relative max-w-4xl mx-auto flex items-end gap-2 bg-slate-900 border border-slate-800 rounded-2xl p-1.5 focus-within:border-blue-500/50 transition-colors">
          <textarea
            ref={inputRef}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Ask about your city..."
            disabled={isTyping}
            rows={1}
            className="flex-1 max-h-32 bg-transparent text-sm text-white placeholder-slate-500 p-3 outline-none resize-none overflow-y-auto"
            style={{ minHeight: '44px' }}
          />
          <button
            onClick={() => handleSend()}
            disabled={!input.trim() || isTyping}
            aria-label="Send message"
            className="mb-0.5 bg-blue-600 disabled:bg-slate-800 hover:bg-blue-500 text-white disabled:text-slate-500 rounded-xl p-3 transition-colors min-h-[44px] min-w-[44px] flex items-center justify-center shrink-0"
          >
            <Send className="w-4 h-4" />
          </button>
        </div>
      </div>
      
    </div>
  );
}