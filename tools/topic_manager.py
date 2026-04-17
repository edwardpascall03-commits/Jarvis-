import os
import json
from datetime import datetime
from collections import deque
from tools.retrieval import store

# Topic classification prompt for Haiku
TOPIC_PROMPT = """Classify this message into exactly one topic category. Respond with ONLY the category name.

Categories:
- CODING: programming, debugging, code, projects, Jarvis development, CTF, cybersecurity technical
- TRAINING: BJJ, Muay Thai, sparring, techniques, gym, martial arts
- UNIVERSITY: coursework, assignments, dissertation, lectures, deadlines
- ADMIN: planning, organisation, scheduling, emails, tasks, productivity
- PERSONAL: wellbeing, goals, habits, relationships, general chat
- RETRIEVAL: asking about past conversations, notes, or previous sessions

Message: {message}

Topic:"""

ROLLING_WINDOW_SIZE = 6

class TopicManager:
    def __init__(self, daily_dir: str):
        self.daily_dir = daily_dir
        self.current_topic = None
        self.session_start = datetime.now().strftime("%H-%M-%S")

        # Rolling window — deque automatically drops oldest when full
        self.rolling_window = deque(maxlen=ROLLING_WINDOW_SIZE)

        # File paths — set when topic is first detected
        self.topic_full_path = None
        self.session_full_path = os.path.join(daily_dir, f"session_{self.session_start}_full.md")

        # All messages this session for the full log
        self.session_messages = []

    def detect_topic(self, message: str) -> str:
        """Use Haiku to classify the topic of a message."""
        try:
            import anthropic
            from dotenv import load_dotenv
            load_dotenv()
            client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
            response = client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=10,
                messages=[{"role": "user", "content": TOPIC_PROMPT.format(message=message)}]
            )
            topic = response.content[0].text.strip().upper()
            valid = ["CODING", "TRAINING", "UNIVERSITY", "ADMIN", "PERSONAL", "RETRIEVAL"]
            return topic if topic in valid else "PERSONAL"
        except Exception as e:
            print(f"[Topic detection error: {e}]")
            return self.current_topic or "PERSONAL"

    def _topic_file_path(self, topic: str) -> str:
        timestamp = datetime.now().strftime("%H-%M-%S")
        return os.path.join(self.daily_dir, f"{topic.lower()}_{timestamp}.md")

    def _write_to_file(self, path: str, role: str, content: str):
        """Append a message to a topic file."""
        with open(path, "a") as f:
            label = "You" if role == "user" else "Jarvis"
            f.write(f"**{label}**: {content}\n\n")

    def _ingest_topic_file(self, path: str, topic: str):
        """Ingest a completed topic file into Chroma and delete it."""
        if not os.path.exists(path):
            return
        with open(path, "r") as f:
            content = f.read().strip()
        if content:
            store(
                content,
                metadata={
                    "source": "topic_session",
                    "topic": topic,
                    "date": datetime.now().strftime("%Y-%m-%d"),
                    "timestamp": datetime.now().isoformat()
                }
            )
        os.remove(path)
        print(f"[Topic file ingested to Chroma: {topic}]")

    def process_message(self, role: str, content: str, message_for_detection: str = None):
        """
        Call this for every user message and every Jarvis reply.
        role: 'user' or 'assistant'
        content: the message text
        message_for_detection: only needed for user messages (what to classify)
        """
        # Detect topic on user messages only
        if role == "user" and message_for_detection:
            # Don't classify very short messages or punctuation
            if len(message_for_detection.strip()) > 15:
                detected = self.detect_topic(message_for_detection)
            else:
                detected = self.current_topic or "PERSONAL"

            if detected != self.current_topic:
                # Topic has changed
                if self.current_topic and self.topic_full_path:
                    # Ingest old topic file to Chroma
                    self._ingest_topic_file(self.topic_full_path, self.current_topic)
                    print(f"[Topic changed: {self.current_topic} → {detected}]")

                # Start new topic file
                self.current_topic = detected
                self.topic_full_path = self._topic_file_path(detected)
                with open(self.topic_full_path, "w") as f:
                    f.write(f"# Topic: {detected} — {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")

        # Write to topic file
        if self.topic_full_path:
            self._write_to_file(self.topic_full_path, role, content)

        # Write to session full log
        self._write_to_file(self.session_full_path, role, content)

        # Update rolling window (user/assistant pairs)
        self.rolling_window.append({"role": role, "content": content})

        # Track for session save
        self.session_messages.append({"role": role, "content": content})

    def get_rolling_context(self) -> str:
        """Return last 6 messages formatted for system prompt injection."""
        if not self.rolling_window:
            return ""
        lines = []
        for msg in self.rolling_window:
            label = "You" if msg["role"] == "user" else "Jarvis"
            lines.append(f"{label}: {msg['content']}")
        return "\n\nRecent conversation context:\n" + "\n".join(lines)

    def close_session(self):
        """Ingest final topic file to Chroma on session end."""
        if self.current_topic and self.topic_full_path:
            self._ingest_topic_file(self.topic_full_path, self.current_topic)
        print(f"[Session log saved: {self.session_full_path}]")