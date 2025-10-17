/**
 * Real-time Competition Scoreboard
 * Shows live human vs AI competition results
 */

'use client';

import React from 'react';
import './CompetitionScoreboard.css';

interface RoundResult {
  round_number: number;
  human: {
    score: number;
    distance_km: number;
    accuracy_rating: string;
    time_taken: number;
  };
  ai: {
    score: number;
    distance_km: number;
    accuracy_rating: string;
    confidence: number;
    reasoning: string;
  };
  comparison: {
    winner: 'human' | 'ai' | 'tie';
    margin: string;
    closer_guess: 'human' | 'ai' | 'tie';
  };
  round_winner: 'human' | 'ai' | 'tie';
}

interface CompetitionScoreboardProps {
  currentRound: number;
  totalRounds: number;
  humanTotalScore: number;
  aiTotalScore: number;
  roundResults: RoundResult[];
  gameActive: boolean;
  className?: string;
}

const CompetitionScoreboard: React.FC<CompetitionScoreboardProps> = ({
  currentRound,
  totalRounds,
  humanTotalScore,
  aiTotalScore,
  roundResults,
  gameActive,
  className = ''
}) => {
  const humanWins = roundResults.filter(r => r.round_winner === 'human').length;
  const aiWins = roundResults.filter(r => r.round_winner === 'ai').length;
  const ties = roundResults.filter(r => r.round_winner === 'tie').length;

  const getOverallLeader = () => {
    if (humanTotalScore > aiTotalScore) return 'human';
    if (aiTotalScore > humanTotalScore) return 'ai';
    return 'tie';
  };

  const formatScore = (score: number) => {
    return score.toLocaleString();
  };

  const getWinnerIcon = (winner: string) => {
    switch (winner) {
      case 'human': return '🧠';
      case 'ai': return '🤖';
      default: return '🤝';
    }
  };

  const getAccuracyColor = (rating: string) => {
    if (rating.includes('Excellent')) return '#4ECDC4';
    if (rating.includes('Very Good')) return '#44A08D';
    if (rating.includes('Good')) return '#95E1D3';
    if (rating.includes('Fair')) return '#FFD93D';
    if (rating.includes('Poor')) return '#FF6B6B';
    return '#DDA0DD';
  };

  return (
    <div className={`competition-scoreboard ${className}`}>
      {/* Main Score Display */}
      <div className="main-score-display">
        <div className={`player-score human-score ${getOverallLeader() === 'human' ? 'leading' : ''}`}>
          <div className="player-icon">🧠</div>
          <div className="score-info">
            <h3>Human Player</h3>
            <div className="total-score">{formatScore(humanTotalScore)}</div>
            <div className="wins-count">{humanWins} rounds won</div>
          </div>
        </div>

        <div className="vs-indicator">
          <div className="vs-text">VS</div>
          <div className="round-indicator">
            Round {currentRound} / {totalRounds}
          </div>
        </div>

        <div className={`player-score ai-score ${getOverallLeader() === 'ai' ? 'leading' : ''}`}>
          <div className="player-icon">🤖</div>
          <div className="score-info">
            <h3>OpenCV AI</h3>
            <div className="total-score">{formatScore(aiTotalScore)}</div>
            <div className="wins-count">{aiWins} rounds won</div>
          </div>
        </div>
      </div>

      {/* Live Status */}
      {gameActive && (
        <div className="live-status">
          <div className="live-indicator">🔴 LIVE</div>
          <div className="status-text">Competition in progress...</div>
        </div>
      )}

      {/* Round History */}
      {roundResults.length > 0 && (
        <div className="round-history">
          <h4>Round Results</h4>
          <div className="rounds-container">
            {roundResults.map((result, index) => (
              <div key={index} className="round-result">
                <div className="round-header">
                  <span className="round-number">Round {result.round_number}</span>
                  <span className="round-winner">
                    {getWinnerIcon(result.round_winner)} 
                    {result.round_winner === 'tie' ? 'Tie' : 
                     result.round_winner === 'human' ? 'Human' : 'AI'} Wins
                  </span>
                </div>
                
                <div className="round-details">
                  <div className="player-result human-result">
                    <div className="player-label">🧠 Human</div>
                    <div className="result-stats">
                      <div className="score-points">{formatScore(result.human.score)} pts</div>
                      <div className="distance">{result.human.distance_km}km away</div>
                      <div 
                        className={`accuracy-rating ${result.human.accuracy_rating.includes('Excellent') ? 'accuracy-excellent' : 
                          result.human.accuracy_rating.includes('Very Good') ? 'accuracy-very-good' :
                          result.human.accuracy_rating.includes('Good') ? 'accuracy-good' :
                          result.human.accuracy_rating.includes('Fair') ? 'accuracy-fair' :
                          result.human.accuracy_rating.includes('Poor') ? 'accuracy-poor' : 'accuracy-very-poor'}`}
                      >
                        {result.human.accuracy_rating}
                      </div>
                      <div className="time-taken">⏱️ {result.human.time_taken}s</div>
                    </div>
                  </div>

                  <div className="player-result ai-result">
                    <div className="player-label">🤖 AI</div>
                    <div className="result-stats">
                      <div className="score-points">{formatScore(result.ai.score)} pts</div>
                      <div className="distance">{result.ai.distance_km}km away</div>
                      <div 
                        className={`accuracy-rating ${result.ai.accuracy_rating.includes('Excellent') ? 'accuracy-excellent' : 
                          result.ai.accuracy_rating.includes('Very Good') ? 'accuracy-very-good' :
                          result.ai.accuracy_rating.includes('Good') ? 'accuracy-good' :
                          result.ai.accuracy_rating.includes('Fair') ? 'accuracy-fair' :
                          result.ai.accuracy_rating.includes('Poor') ? 'accuracy-poor' : 'accuracy-very-poor'}`}
                      >
                        {result.ai.accuracy_rating}
                      </div>
                      <div className="confidence">🎯 {(result.ai.confidence * 100).toFixed(0)}% confident</div>
                    </div>
                  </div>
                </div>

                <div className="round-comparison">
                  <div className="margin-info">{result.comparison.margin}</div>
                  {result.ai.reasoning && (
                    <div className="ai-reasoning">
                      <strong>AI Reasoning:</strong> {result.ai.reasoning}
                    </div>
                  )}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Quick Stats */}
      {roundResults.length > 0 && (
        <div className="quick-stats">
          <div className="stat-item">
            <span className="stat-label">Rounds Played</span>
            <span className="stat-value">{roundResults.length}</span>
          </div>
          <div className="stat-item">
            <span className="stat-label">Human Wins</span>
            <span className="stat-value human-color">{humanWins}</span>
          </div>
          <div className="stat-item">
            <span className="stat-label">AI Wins</span>
            <span className="stat-value ai-color">{aiWins}</span>
          </div>
          <div className="stat-item">
            <span className="stat-label">Ties</span>
            <span className="stat-value">{ties}</span>
          </div>
        </div>
      )}
    </div>
  );
};

export default CompetitionScoreboard;
