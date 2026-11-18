"""База данных: модели по схеме проекта и операции записи (PostgreSQL, подключение через DB_URL)."""

from sqlalchemy import (
    BigInteger,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Identity,
    String,
    Text,
    create_engine,
    func,
    select,
)
from sqlalchemy.orm import DeclarativeBase, Session

# Идентификатор pgserver-инстанса передаётся драйвером psycopg2
PSYCOPG2_URL_PREFIX = "postgresql+psycopg2://"


class Base(DeclarativeBase):
    pass


class ModelVersion(Base):
    __tablename__ = "model_versions"

    # Identity = автоинкремент средствами PostgreSQL, id из кода не подставляем
    id = Column(BigInteger, Identity(), primary_key=True)
    version = Column(String(50), nullable=False, unique=True)
    model_name = Column(String(100), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class ClassificationResult(Base):
    __tablename__ = "classification_results"

    id = Column(BigInteger, Identity(), primary_key=True)
    text = Column(Text, nullable=False)
    predicted_class = Column(String(20), nullable=False)
    true_class = Column(String(20), nullable=True)  # заполняется при ручной проверке
    confidence = Column(Float, nullable=False)
    model_version_id = Column(BigInteger, ForeignKey("model_versions.id"), nullable=False)
    source = Column(String(20), nullable=False)  # telegram / api
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class Database:
    def __init__(self, url: str):
        if url.startswith("postgresql://"):
            url = PSYCOPG2_URL_PREFIX + url.removeprefix("postgresql://")
        self._engine = create_engine(url)

    def init(self) -> None:
        Base.metadata.create_all(self._engine)

    def get_or_create_model_version(self, model_name: str, version: str) -> int:
        with Session(self._engine) as s:
            row = s.scalar(select(ModelVersion).where(ModelVersion.version == version))
            if row is None:
                row = ModelVersion(version=version, model_name=model_name)
                s.add(row)
            s.commit()
            return row.id

    def save_prediction(
        self,
        text: str,
        predicted_class: str,
        confidence: float,
        model_version_id: int,
        source: str,
    ) -> None:
        with Session(self._engine) as s:
            s.add(
                ClassificationResult(
                    text=text,
                    predicted_class=predicted_class,
                    confidence=confidence,
                    model_version_id=model_version_id,
                    source=source,
                )
            )
            s.commit()

    def model_versions(self) -> list[ModelVersion]:
        with Session(self._engine) as s:
            return list(s.scalars(select(ModelVersion)))

    def recent_predictions(self, limit: int = 10) -> list[ClassificationResult]:
        with Session(self._engine) as s:
            return list(s.scalars(select(ClassificationResult).order_by(ClassificationResult.id.desc()).limit(limit)))

    def close(self) -> None:
        self._engine.dispose()
