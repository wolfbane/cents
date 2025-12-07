"""Research API routes."""

from flask import Blueprint, jsonify, request

from cents.agents import AGENTS, OrchestratorAgent
from cents.db import ThesisRepository, EvidenceRepository
from cents.serialization import serialize

from .errors import ValidationError, NotFoundError

research_bp = Blueprint("research", __name__)


@research_bp.route("/research/<symbol>", methods=["POST"])
def run_research(symbol: str):
    """Run research agents on a symbol.

    Query params:
        agent: Specific agent to run (optional, defaults to orchestrator)
        save: Whether to save evidence (optional, default true)
        thesis_id: Thesis to evaluate against (optional)

    Returns:
        Research results with evidence and conviction delta
    """
    symbol = symbol.upper()

    # Get optional parameters
    agent_name = request.args.get("agent")
    save = request.args.get("save", "true").lower() != "false"
    thesis_id = request.args.get("thesis_id")

    # Get thesis if specified
    thesis = None
    if thesis_id:
        thesis_repo = ThesisRepository()
        thesis = thesis_repo.get(thesis_id)
        if thesis is None:
            raise NotFoundError(f"Thesis {thesis_id} not found")

    # Validate agent name if specified
    if agent_name and agent_name not in AGENTS:
        raise ValidationError(
            f"Unknown agent: {agent_name}",
            details={"valid_agents": list(AGENTS.keys())}
        )

    # Run agent(s)
    if agent_name:
        agent = AGENTS[agent_name]()
    else:
        agent = OrchestratorAgent()

    result = agent.research(symbol, thesis)

    # Save evidence if requested
    evidence_count = 0
    if save and result.evidence:
        evidence_repo = EvidenceRepository()
        for e in result.evidence:
            e.symbol = symbol
            if thesis:
                e.thesis_id = thesis.id
            if evidence_repo.create(e, dedupe=True):
                evidence_count += 1

        # Update thesis conviction if linked
        if thesis and evidence_count > 0:
            thesis.update_conviction(result.conviction_delta)
            ThesisRepository().update(thesis)

    return jsonify({
        "symbol": symbol,
        "thesis_id": thesis.id if thesis else None,
        "conviction_delta": result.conviction_delta,
        "summary": result.summary,
        "dimension_scores": result.dimension_scores,
        "evidence": [serialize(e) for e in result.evidence],
        "evidence_saved": evidence_count,
    })


@research_bp.route("/research/agents", methods=["GET"])
def list_agents():
    """List available research agents."""
    return jsonify({
        "agents": list(AGENTS.keys())
    })
