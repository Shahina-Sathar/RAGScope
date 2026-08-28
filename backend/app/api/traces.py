from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.session import get_session
from app.models.schemas import Trace, TraceResponse, TraceListResponse

router = APIRouter(prefix="/traces", tags=["traces"])


@router.get("", response_model=list[TraceListResponse])
async def list_traces(
    limit: int = 20,
    offset: int = 0,
    session: AsyncSession = Depends(get_session),
):
    result = await session.execute(
        select(Trace)
        .order_by(Trace.created_at.desc())
        .offset(offset)
        .limit(limit)
    )
    traces = result.scalars().all()

    return [
        TraceListResponse(
            id=trace.id,
            original_query=trace.original_query,
            total_latency_ms=trace.latency.get("total_ms", 0),
            num_sources=len(trace.retrieval_results.get("chunks", [])),
            created_at=trace.created_at,
        )
        for trace in traces
    ]


@router.get("/{trace_id}", response_model=TraceResponse)
async def get_trace(
    trace_id: str,
    session: AsyncSession = Depends(get_session),
):
    result = await session.execute(select(Trace).where(Trace.id == trace_id))
    trace = result.scalar_one_or_none()

    if not trace:
        raise HTTPException(status_code=404, detail="Trace not found")

    return trace
