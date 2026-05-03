"""Database models for SQLModel ORM."""
from typing import Optional
from sqlmodel import Field, Column, SQLModel, SmallInteger
from sqlalchemy import TIMESTAMP, CheckConstraint, Numeric
from sqlalchemy.dialects import postgresql as pg
from uuid import UUID, uuid4
from datetime import datetime, date, timezone


class EarthquakeRaw(SQLModel, table=True):
    """Raw earthquake events ingested from external sources (DIBI/BMKG).

    Stored with location and casualty fields for downstream analysis.
    """
    # Table Name
    __tablename__ = "dibi_earthquakes_raw"

    # Primary Key
    uid: Optional[UUID] = Field(
        default_factory=uuid4,
        sa_column=Column(pg.UUID(as_uuid=True), primary_key=True, unique=True),
    )
    # Data Columns
    kode_identitas_bencana: str = Field(nullable=False)
    id_kabupaten: int = Field(nullable=False)
    tanggal_kejadian: date = Field(nullable=False)
    lokasi: Optional[str] = None
    kabupaten: str = Field(nullable=False)
    provinsi: str = Field(nullable=False)

    korban_meninggal: int = Field(default=0)
    korban_hilang: int = Field(default=0)
    korban_terluka: int = Field(default=0)
    rumah_rusak: int = Field(default=0)
    rumah_terendam: int = Field(default=0)
    fasum_rusak: int = Field(default=0)

    # Representation Method
    def __repr__(self):
        return f"<EarthquakeRaw => {self.uid} | {self.kabupaten}, {self.provinsi}>"


class RealtimePredict(SQLModel, table=True):
    """Realtime model predictions stored for recent BMKG events."""
    __tablename__ = "realtime_predictions"

    __table_args__ = (
        CheckConstraint("magnitude > 0", name="chk_predict_magnitude"),
        CheckConstraint("depth >= 0", name="chk_predict_depth"),
        CheckConstraint("latitude BETWEEN -90 AND 90", name="chk_predict_latitude"),
        CheckConstraint("longitude BETWEEN -180 AND 180", name="chk_predict_longitude"),
        CheckConstraint("risk_score BETWEEN 0 AND 100", name="chk_predict_risk_score"),
        CheckConstraint("cluster >= 0", name="chk_predict_cluster"),
    )

    id: Optional[UUID] = Field(
        sa_column=Column(pg.UUID(as_uuid=True), primary_key=True, default=uuid4)
    )

    # --- Data Gempa ---
    tanggal: Optional[str] = Field(default=None, max_length=20)
    jam: Optional[str] = Field(default=None, max_length=20)
    koordinat: Optional[str] = Field(default=None, max_length=50)
    latitude: Optional[float] = Field(default=None, sa_column=Column(Numeric(9, 6)))
    longitude: Optional[float] = Field(default=None, sa_column=Column(Numeric(9, 6)))
    magnitude: Optional[float] = Field(default=None, sa_column=Column(Numeric(4, 1)))
    depth: Optional[float] = Field(default=None, sa_column=Column(Numeric(7, 3)))
    wilayah: Optional[str] = None
    cluster: Optional[int] = Field(default=None, sa_column=Column(SmallInteger()))
    risk_score: Optional[float] = Field(default=None, sa_column=Column(Numeric(5, 2)))
    risk_level: Optional[str] = Field(default=None, max_length=20)

    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        sa_column=Column(TIMESTAMP(timezone=True)),  # TIMESTAMPTZ
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        sa_column=Column(TIMESTAMP(timezone=True)),  # TIMESTAMPTZ
    )

    def __repr__(self):
        return (
            f"<RealtimePredict => {self.id} | "
            f"M{self.magnitude} | cluster: {self.cluster} | "
            f"{self.risk_level} ({self.risk_score})>"
        )


class RegionRiskStatic(SQLModel, table=True):
    """Static per-region risk metrics and ML results used for choropleth."""
    __tablename__ = "region_risk_static"

    __table_args__ = (
        CheckConstraint("risk_score >= 0", name="chk_static_risk_score_min"),
        CheckConstraint("cluster_label >= 0", name="chk_static_cluster"),
    )

    # --- Primary Key ---
    id: Optional[UUID] = Field(
        sa_column=Column(pg.UUID(as_uuid=True), primary_key=True, default=uuid4)
    )

    # --- Identitas Wilayah ---
    id_kabupaten: str = Field(nullable=False, index=True, max_length=20)
    nama_kabupaten: str = Field(nullable=False, index=True, max_length=100)

    # --- Fitur (Numerik → pakai Numeric biar presisi) ---
    luas_wilayah_km2: Optional[float] = Field(
        default=None, sa_column=Column(Numeric(10, 4))
    )
    frekuensi_gempa: Optional[float] = Field(
        default=None, sa_column=Column(Numeric(10, 6))
    )
    mag_max: Optional[float] = Field(default=None, sa_column=Column(Numeric(4, 2)))
    mag_mean: Optional[float] = Field(default=None, sa_column=Column(Numeric(4, 2)))
    depth_mean: Optional[float] = Field(default=None, sa_column=Column(Numeric(6, 2)))

    korban_total: Optional[float] = Field(
        default=None, sa_column=Column(Numeric(12, 2))
    )
    rumah_rusak_total: Optional[float] = Field(
        default=None, sa_column=Column(Numeric(12, 2))
    )
    fasum_rusak_total: Optional[float] = Field(
        default=None, sa_column=Column(Numeric(12, 2))
    )

    # --- Hasil ML ---
    cluster_label: Optional[int] = Field(default=None, sa_column=Column(pg.SMALLINT))
    risk_score: Optional[float] = Field(default=None, sa_column=Column(Numeric(5, 4)))
    risk_level: Optional[str] = Field(default=None, max_length=20)

    # --- PCA (opsional visualisasi) ---
    PC1: Optional[float] = Field(default=None, sa_column=Column(Numeric(10, 6)))
    PC2: Optional[float] = Field(default=None, sa_column=Column(Numeric(10, 6)))

    # --- Timestamp ---
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        sa_column=Column(TIMESTAMP(timezone=True)),
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        sa_column=Column(TIMESTAMP(timezone=True)),
    )

    # --- Representation ---
    def __repr__(self):
        return (
            f"<RegionRiskStatic => {self.nama_kabupaten} | "
            f"cluster: {self.cluster_label} | "
            f"{self.risk_level} ({self.risk_score})>"
        )
