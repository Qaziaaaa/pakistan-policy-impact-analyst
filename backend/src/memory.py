import os
from datetime import datetime
from dotenv import load_dotenv
from sqlalchemy import (
    create_engine,
    Column,
    String,
    Text,
    DateTime,
    Integer,
    ForeignKey
)
from sqlalchemy.orm import declarative_base, sessionmaker, relationship

load_dotenv("backend/.env")

DATABASE_URL = os.getenv("DATABASE_URL").replace("postgresql://", "postgresql+psycopg2://")
engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class UserProfile(Base):
    """Stores user persona, research domain, and saved preferences across sessions."""
    __tablename__ = "user_profiles"

    user_id = Column(String(100), primary_key=True)
    role = Column(String(100), default="General Researcher") # e.g. IMF Analyst, Student, Journalist
    preferred_topics = Column(Text, default="Macroeconomics, Inflation, Monetary Policy")
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    findings = relationship("ResearchFinding", back_populates="user", cascade="all, delete-orphan")

class ResearchFinding(Base):
    """Stores verified economic facts and conclusions saved by the research assistant."""
    __tablename__ = "research_findings"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String(100), ForeignKey("user_profiles.user_id"), nullable=False)
    topic = Column(String(200), nullable=False)
    finding = Column(Text, nullable=False)
    citation = Column(String(300), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    user = relationship("UserProfile", back_populates="findings")

# Create tables
Base.metadata.create_all(bind=engine)

class LongTermMemory:
    """Helper API for agents to read and persist user context and findings."""
    def __init__(self):
        self.db = SessionLocal()

    def get_or_create_user(self, user_id: str, role: str = "Economic Analyst") -> UserProfile:
        user = self.db.query(UserProfile).filter(UserProfile.user_id == user_id).first()
        if not user:
            user = UserProfile(user_id=user_id, role=role)
            self.db.add(user)
            self.db.commit()
            self.db.refresh(user)
        return user

    def update_user_preferences(self, user_id: str, role: str = None, preferred_topics: str = None):
        user = self.get_or_create_user(user_id)
        if role:
            user.role = role
        if preferred_topics:
            user.preferred_topics = preferred_topics
        self.db.commit()
        self.db.refresh(user)
        return user

    def save_finding(self, user_id: str, topic: str, finding: str, citation: str = ""):
        self.get_or_create_user(user_id)
        entry = ResearchFinding(
            user_id=user_id,
            topic=topic,
            finding=finding,
            citation=citation
        )
        self.db.add(entry)
        self.db.commit()
        self.db.refresh(entry)
        return entry

    def get_user_memory_summary(self, user_id: str) -> str:
        """Returns a concise summary of user preferences and past findings for LLM context."""
        user = self.get_or_create_user(user_id)
        findings = self.db.query(ResearchFinding).filter(
            ResearchFinding.user_id == user_id
        ).order_by(ResearchFinding.created_at.desc()).limit(5).all()

        summary = f"User Role: {user.role}\nPreferences: {user.preferred_topics}\n"
        if findings:
            summary += "Past Saved Insights:\n"
            for f in findings:
                summary += f"- [{f.topic}]: {f.finding} (Ref: {f.citation})\n"
        return summary.strip()

    def close(self):
        self.db.close()

if __name__ == "__main__":
    memory = LongTermMemory()
    test_user_id = "analyst_qazi_01"

    print(f"🧠 Testing Long-Term Memory for '{test_user_id}'...")
    memory.update_user_preferences(
        user_id=test_user_id,
        role="Senior Macroeconomic Strategist",
        preferred_topics="Inflation, SBP Policy Rates, Debt Servicing"
    )

    memory.save_finding(
        user_id=test_user_id,
        topic="Inflation FY24",
        finding="Headline CPI dropped from 38% in May 2023 to 11.8% in May 2024.",
        citation="Pakistan Economic Survey 2023-24, Page 1"
    )

    context = memory.get_user_memory_summary(test_user_id)
    print("\n✅ Retrieved Long-Term Memory Context:\n" + "="*40)
    print(context)
    print("="*40)
    memory.close()
