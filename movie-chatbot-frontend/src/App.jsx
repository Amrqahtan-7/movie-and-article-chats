import { useState, useRef, useEffect } from "react";
import "./App.css";

const BASE_URL = "http://127.0.0.1:8000";

function App() {
  // ---- Auth state ----
  const [token, setToken] = useState(localStorage.getItem("token") || null);
  const [userEmail, setUserEmail] = useState(localStorage.getItem("userEmail") || "");
  const [authMode, setAuthMode] = useState("login"); // "login" or "signup"
  const [authEmail, setAuthEmail] = useState("");
  const [authPassword, setAuthPassword] = useState("");
  const [authError, setAuthError] = useState("");

  // ---- View switcher: "chat" or "article" ----
  const [view, setView] = useState("chat");

  // ---- Chat state ----
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [sessionId, setSessionId] = useState(null);
  const [sessions, setSessions] = useState([]);
  const chatEndRef = useRef(null);
  const fileInputRef = useRef(null);

  // ---- Article generator state ----
  const [articleTopic, setArticleTopic] = useState("");
  const [articleLoading, setArticleLoading] = useState(false);
  const [articleResult, setArticleResult] = useState(null); // { status, article, saved_path?, filename?, pending? }
  const [articleError, setArticleError] = useState("");
  const [articleSuccess, setArticleSuccess] = useState(""); // shown after a confirmed save
  const [articleHistory, setArticleHistory] = useState([]);
  const [revisionInput, setRevisionInput] = useState("");
  const [revising, setRevising] = useState(false);
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  useEffect(() => {
    if (token) loadSessions();
  }, [token]);

  useEffect(() => {
    if (token && view === "article") loadArticleHistory();
  }, [token, view]);

  // ---- Helper: fetch with the auth header attached ----
  const authFetch = (url, options = {}) => {
    return fetch(url, {
      ...options,
      headers: {
        ...(options.headers || {}),
        Authorization: `Bearer ${token}`,
      },
    });
  };

  // ---- Auth actions ----
  const handleAuthSubmit = async () => {
    setAuthError("");
    const endpoint = authMode === "login" ? "/login" : "/signup";

    try {
      const res = await fetch(`${BASE_URL}${endpoint}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ email: authEmail, password: authPassword }),
      });
      const data = await res.json();

      if (!res.ok) {
        setAuthError(data.detail || "Something went wrong");
        return;
      }

      localStorage.setItem("token", data.access_token);
      localStorage.setItem("userEmail", data.email);
      setToken(data.access_token);
      setUserEmail(data.email);
    } catch (error) {
      setAuthError("Could not connect to server");
    }
  };

  const logout = () => {
    localStorage.removeItem("token");
    localStorage.removeItem("userEmail");
    setToken(null);
    setUserEmail("");
    setMessages([]);
    setSessionId(null);
    setSessions([]);
    setArticleResult(null);
  };

  // ---- Chat actions ----
  const loadSessions = async () => {
    try {
      const res = await authFetch(`${BASE_URL}/sessions`);
      const data = await res.json();
      setSessions(data);
    } catch (error) {
      console.error("Failed to load sessions", error);
    }
  };

  const openSession = async (id) => {
    try {
      const res = await authFetch(`${BASE_URL}/sessions/${id}`);
      const data = await res.json();
      setMessages(
        data.map((m) => ({
          sender: m.role === "user" ? "user" : "bot",
          text: m.content,
        }))
      );
      setSessionId(id);
    } catch (error) {
      console.error("Failed to load chat", error);
    }
  };

  const startNewChat = () => {
    setMessages([]);
    setSessionId(null);
    setView("chat");
  };

  const deleteSession = async (id, e) => {
    e.stopPropagation();
    if (!window.confirm("Delete this chat?")) return;

    try {
      await authFetch(`${BASE_URL}/sessions/${id}`, { method: "DELETE" });
      if (id === sessionId) {
        setMessages([]);
        setSessionId(null);
      }
      loadSessions();
    } catch (error) {
      console.error("Failed to delete session", error);
    }
  };

  const sendMessage = async () => {
    const trimmed = input.trim();
    if (!trimmed) return;

    const newMessages = [...messages, { sender: "user", text: trimmed }];
    setMessages(newMessages);
    setInput("");
    setLoading(true);

    try {
      const response = await authFetch(`${BASE_URL}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: trimmed, session_id: sessionId }),
      });
      const data = await response.json();

      setMessages([...newMessages, { sender: "bot", text: data.answer }]);
      setSessionId(data.session_id);
      loadSessions();
    } catch (error) {
      setMessages([
        ...newMessages,
        { sender: "bot", text: "Error connecting to server." },
      ]);
    } finally {
      setLoading(false);
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === "Enter") sendMessage();
  };

  const handleFileUpload = async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    setUploading(true);
    setMessages((prev) => [
      ...prev,
      { sender: "bot", text: `Uploading "${file.name}"...` },
    ]);

    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await authFetch(`${BASE_URL}/upload`, {
        method: "POST",
        body: formData,
      });
      const data = await response.json();

      const resultText = data.error
        ? `Upload failed: ${data.error}`
        : `"${data.filename}" saved and added (${data.chunks_added} chunks).`;

      setMessages((prev) => [...prev, { sender: "bot", text: resultText }]);
    } catch (error) {
      setMessages((prev) => [
        ...prev,
        { sender: "bot", text: "Error uploading file." },
      ]);
    } finally {
      setUploading(false);
      e.target.value = "";
    }
  };

  // ---- Article generator actions ----
  const loadArticleHistory = async () => {
    try {
      const res = await authFetch(`${BASE_URL}/articles`);
      const data = await res.json();
      setArticleHistory(data);
    } catch (error) {
      console.error("Failed to load article history", error);
    }
  };

  const openArticle = async (filename) => {
    setArticleError("");
    setArticleSuccess("");
    try {
      const res = await authFetch(`${BASE_URL}/articles/${filename}`);
      const data = await res.json();
      setArticleResult(data);
      setArticleTopic("");
    } catch (error) {
      setArticleError("Failed to load that article.");
    }
  };

  const deleteArticle = async (filename, e) => {
    e.stopPropagation();
    if (!window.confirm("Delete this article?")) return;

    try {
      await authFetch(`${BASE_URL}/articles/${filename}`, { method: "DELETE" });
      loadArticleHistory();
    } catch (error) {
      console.error("Failed to delete article", error);
    }
  };

  const generateArticle = async () => {
    const trimmed = articleTopic.trim();
    if (!trimmed) return;

    setArticleLoading(true);
    setArticleError("");
    setArticleSuccess("");
    setArticleResult(null);

    try {
      const response = await authFetch(`${BASE_URL}/generate-article`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ topic: trimmed }),
      });

      if (!response.ok) {
        const errData = await response.json().catch(() => ({}));
        setArticleError(errData.detail || "Failed to generate article.");
        return;
      }

      const data = await response.json();
      setArticleResult(data);
      loadArticleHistory();
    } catch (error) {
      setArticleError("Could not connect to server.");
    } finally {
      setArticleLoading(false);
    }
  };

  const handleArticleKeyPress = (e) => {
    if (e.key === "Enter" && !articleLoading) generateArticle();
  };

  const reviseArticleDraft = async () => {
    const trimmed = revisionInput.trim();
    if (!trimmed) return;

    setRevising(true);
    setArticleError("");

    try {
      const response = await authFetch(`${BASE_URL}/article/revise`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ instruction: trimmed }),
      });

      if (!response.ok) {
        const errData = await response.json().catch(() => ({}));
        setArticleError(errData.detail || "Failed to revise the article.");
        return;
      }

      const data = await response.json();
      setArticleResult(data);
      setRevisionInput("");
    } catch (error) {
      setArticleError("Could not connect to server.");
    } finally {
      setRevising(false);
    }
  };

  const confirmArticleSave = async () => {
    setSaving(true);
    setArticleError("");
    setArticleSuccess("");

    try {
      const response = await authFetch(`${BASE_URL}/article/confirm`, { method: "POST" });

      if (!response.ok) {
        const errData = await response.json().catch(() => ({}));
        setArticleError(errData.detail || "Failed to save the article.");
        return;
      }

      const data = await response.json();

      // Saved: confirm it and clear the panel for a new article in one step,
      // so the success line is what's left on screen instead of the old draft.
      setArticleSuccess(`✔ Saved as "${data.filename}" — it's in your history on the left.`);
      setArticleResult(null);
      setArticleTopic("");
      setRevisionInput("");
      loadArticleHistory();
    } catch (error) {
      setArticleError("Could not connect to server.");
    } finally {
      setSaving(false);
    }
  };

  const discardArticleDraft = async () => {
    if (!window.confirm("Discard this draft without saving?")) return;
    try {
      await authFetch(`${BASE_URL}/article/discard`, { method: "POST" });
    } catch (error) {
      console.error("Failed to discard", error);
    } finally {
      setArticleResult(null);
    }
  };

  const handleRevisionKeyPress = (e) => {
    if (e.key === "Enter" && !revising) reviseArticleDraft();
  };

  // ---- Render: login/signup screen if not authenticated ----
  if (!token) {
    return (
      <div className="auth-screen">
        <div className="auth-box">
          <h1>🎬 Movie Chatbot</h1>
          <p className="tagline">
            {authMode === "login" ? "Welcome back" : "Create your account"}
          </p>

          <input
            type="email"
            placeholder="Email"
            value={authEmail}
            onChange={(e) => setAuthEmail(e.target.value)}
          />
          <input
            type="password"
            placeholder="Password"
            value={authPassword}
            onChange={(e) => setAuthPassword(e.target.value)}
          />

          {authError && <div className="auth-error">{authError}</div>}

          <button onClick={handleAuthSubmit}>
            {authMode === "login" ? "Log In" : "Sign Up"}
          </button>

          <p className="auth-switch">
            {authMode === "login" ? (
              <>
                Don't have an account?{" "}
                <span onClick={() => setAuthMode("signup")}>Sign up</span>
              </>
            ) : (
              <>
                Already have an account?{" "}
                <span onClick={() => setAuthMode("login")}>Log in</span>
              </>
            )}
          </p>
        </div>
      </div>
    );
  }

  // ---- Render: main app if authenticated ----
  return (
    <div className="layout">
      <div className="sidebar">
        <button
          className="new-chat-btn"
          onClick={() => {
            if (view === "chat") {
              startNewChat();
            } else {
              setArticleResult(null);
              setArticleTopic("");
              setArticleError("");
              setArticleSuccess("");
            }
          }}
        >
          {view === "chat" ? "+ New Chat" : "+ New Article"}
        </button>

        <div className="view-toggle">
          <button
            className={view === "chat" ? "active" : ""}
            onClick={() => setView("chat")}
          >
            Chat
          </button>
          <button
            className={view === "article" ? "active" : ""}
            onClick={() => setView("article")}
          >
            Generate Article
          </button>
        </div>

        {view === "chat" && (
          <div className="session-list">
            {sessions.map((s) => (
              <div
                key={s.id}
                className={`session-item ${s.id === sessionId ? "active" : ""}`}
                onClick={() => openSession(s.id)}
              >
                <span className="session-title">{s.title}</span>
                <button
                  className="delete-btn"
                  onClick={(e) => deleteSession(s.id, e)}
                  title="Delete chat"
                >
                  🗑
                </button>
              </div>
            ))}
          </div>
        )}

        {view === "article" && (
          <div className="session-list">
            {articleHistory.map((a) => (
              <div
                key={a.filename}
                className={`session-item ${articleResult?.filename === a.filename ? "active" : ""}`}
                onClick={() => openArticle(a.filename)}
              >
                <span className={`history-dot ${a.status}`} />
                <span className="session-title">{a.topic}</span>
                <button
                  className="delete-btn"
                  onClick={(e) => deleteArticle(a.filename, e)}
                  title="Delete article"
                >
                  🗑
                </button>
              </div>
            ))}
            {articleHistory.length === 0 && (
              <div className="empty-history">No articles yet</div>
            )}
          </div>
        )}

        <div className="user-footer">
          <span className="user-email">{userEmail}</span>
          <button className="logout-btn" onClick={logout}>
            Log out
          </button>
        </div>
      </div>

      {view === "chat" ? (
        <div className="app">
          <h1>🎬 Movie Chatbot</h1>
          <p className="tagline">Ask the archive anything</p>

          <div className="chat-box">
            <div className="message-list">
              {messages.map((msg, i) => (
                <div key={i} className={`message ${msg.sender}`}>
                  {msg.text}
                </div>
              ))}
              {loading && <div className="message bot">Thinking...</div>}
              <div ref={chatEndRef} />
            </div>
          </div>

          <div className="input-area">
            <input
              type="file"
              accept=".pdf,.docx,.txt"
              ref={fileInputRef}
              onChange={handleFileUpload}
              style={{ display: "none" }}
            />
            <button
              type="button"
              className="upload-btn"
              onClick={() => fileInputRef.current.click()}
              disabled={uploading}
              title="Upload PDF, Word, or TXT"
            >
              📎
            </button>

            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyPress={handleKeyPress}
              placeholder="Ask about a movie or your uploaded file..."
            />
            <button onClick={sendMessage} disabled={loading}>
              Send
            </button>
          </div>
        </div>
      ) : (
        <div className="app">
          <h1>📝 Article Generator</h1>
          <p className="tagline">Research → write → verify, automatically</p>

          <div className="article-panel">
            <div className="article-form">
              <input
                type="text"
                value={articleTopic}
                onChange={(e) => setArticleTopic(e.target.value)}
                onKeyPress={handleArticleKeyPress}
                placeholder="Enter a topic, e.g. the history of stop-motion animation"
                disabled={articleLoading}
              />
              <button onClick={generateArticle} disabled={articleLoading || !articleTopic.trim()}>
                {articleLoading ? "Generating..." : "Generate"}
              </button>
            </div>

            {articleLoading && (
              <div className="article-status">
                Researching, writing, and verifying — this can take a little
                longer than a normal chat reply, especially if it needs to
                re-check its sources.
              </div>
            )}

            {articleError && <div className="auth-error">{articleError}</div>}

            {articleSuccess && <div className="article-success">{articleSuccess}</div>}

            {articleResult && (
              <div className="article-result">
                <div className={`status-badge ${articleResult.status}`}>
                  {articleResult.status === "verified"
                    ? "✓ Verified against sources"
                    : "⚠ Unverified — treat with caution"}
                </div>

                <div className="article-text">
                  {articleResult.article.split("\n").map((line, i) => (
                    <p key={i}>{line}</p>
                  ))}
                </div>

                {articleResult.pending ? (
                  <>
                    <div className="revision-box">
                      <input
                        type="text"
                        value={revisionInput}
                        onChange={(e) => setRevisionInput(e.target.value)}
                        onKeyPress={handleRevisionKeyPress}
                        placeholder="Ask for a change, e.g. 'make the intro shorter'"
                        disabled={revising}
                      />
                      <button onClick={reviseArticleDraft} disabled={revising || !revisionInput.trim()}>
                        {revising ? "Revising..." : "Revise"}
                      </button>
                    </div>
                    <div className="article-actions">
                      <button className="save-btn" onClick={confirmArticleSave} disabled={saving}>
                        {saving ? "Saving..." : "✔ Confirm & Save"}
                      </button>
                      <button className="discard-btn" onClick={discardArticleDraft} disabled={saving}>
                        Discard
                      </button>
                    </div>
                  </>
                ) : (
                  <div className="article-path">
                    Saved to: <code>{articleResult.saved_path}</code>
                  </div>
                )}
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

export default App;