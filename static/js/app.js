/**
 * Frontend JavaScript for Indonesian Legal QA Web Application
 * Handles SSE events, UI updates, and process visualization
 */

class LegalQAApp {
    constructor() {
        this.sessionId = null;
        this.eventSource = null;
        this.isProcessing = false;
        
        // Data storage
        this.graphragIterations = [];
        this.naiveragIterations = [];
        this.amendmentData = {
            graphrag: null,
            naiverag: null,
            aggregated: null
        };
        this.aggregatorResult = null;
        this.finalAnswer = null;
        this.currentMode = 'before_aggregation';
        
        this.init();
    }
    
    init() {
        // Bind event handlers
        document.getElementById('submit-btn').addEventListener('click', () => this.submitQuestion());
        document.getElementById('question-input').addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && e.ctrlKey) {
                this.submitQuestion();
            }
        });
        
        // Tab switching
        document.querySelectorAll('.tab-btn').forEach(btn => {
            btn.addEventListener('click', (e) => this.switchTab(e.target.dataset.tab));
        });
        
        // Mode selector
        document.querySelectorAll('input[name="mode"]').forEach(radio => {
            radio.addEventListener('change', (e) => {
                this.currentMode = e.target.value;
            });
        });
    }
    
    generateSessionId() {
        return `session_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    }
    
    async submitQuestion() {
        const questionInput = document.getElementById('question-input');
        const question = questionInput.value.trim();
        
        if (!question) {
            alert('Please enter a question.');
            return;
        }
        
        if (this.isProcessing) {
            return;
        }
        
        // Reset state
        this.resetState();
        this.isProcessing = true;
        this.sessionId = this.generateSessionId();
        
        // Update UI
        this.setButtonLoading(true);
        this.showStatusBanner('Initializing pipeline...', 'processing');
        this.showResultsSection();
        
        // Update mode description
        const modeDesc = document.getElementById('amendment-mode-desc');
        if (this.currentMode === 'before_aggregation') {
            modeDesc.textContent = 'Mode: Before Aggregation - Amendments applied to each pipeline answer before combining';
        } else {
            modeDesc.textContent = 'Mode: After Aggregation - Amendments applied to the final aggregated answer';
        }
        
        try {
            // Start SSE connection
            this.startEventStream();
            
            // Submit query
            const response = await fetch('/api/query', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    question: question,
                    mode: this.currentMode,
                    session_id: this.sessionId
                })
            });
            
            if (!response.ok) {
                throw new Error(`HTTP error: ${response.status}`);
            }
            
            const data = await response.json();
            console.log('Query submitted:', data);
            
        } catch (error) {
            console.error('Error submitting question:', error);
            this.showStatusBanner(`Error: ${error.message}`, 'error');
            this.setButtonLoading(false);
            this.isProcessing = false;
        }
    }
    
    startEventStream() {
        if (this.eventSource) {
            this.eventSource.close();
        }
        
        this.eventSource = new EventSource(`/api/events/${this.sessionId}`);
        
        this.eventSource.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);
                this.handleEvent(data);
            } catch (e) {
                console.error('Error parsing event:', e);
            }
        };
        
        this.eventSource.onerror = (error) => {
            console.error('EventSource error:', error);
            if (this.isProcessing) {
                this.showStatusBanner('Connection lost. Please try again.', 'error');
            }
            this.eventSource.close();
        };
    }
    
    handleEvent(event) {
        console.log('Event received:', event.type, event.data);
        
        switch (event.type) {
            case 'status':
                this.handleStatusEvent(event.data);
                break;
            case 'language_detected':
                this.showStatusBanner(`Language detected: ${event.data.language === 'id' ? 'Indonesian' : 'English'}`, 'processing');
                break;
            case 'pipeline_start':
                this.handlePipelineStart(event.data);
                break;
            case 'iteration_start':
                this.handleIterationStart(event.data);
                break;
            case 'retrieval_complete':
                this.handleRetrievalComplete(event.data);
                break;
            case 'answer_generated':
                this.handleAnswerGenerated(event.data);
                break;
            case 'judge_decision':
                this.handleJudgeDecision(event.data);
                break;
            case 'query_modified':
                this.handleQueryModified(event.data);
                break;
            case 'iteration_complete':
                this.handleIterationComplete(event.data);
                break;
            case 'pipeline_complete':
                this.handlePipelineComplete(event.data);
                break;
            case 'pipeline_error':
                this.handlePipelineError(event.data);
                break;
            case 'amendment_start':
                this.handleAmendmentStart(event.data);
                break;
            case 'uu_references_extracted':
                this.handleUUReferencesExtracted(event.data);
                break;
            case 'amendment_chains_retrieved':
                this.handleAmendmentChainsRetrieved(event.data);
                break;
            case 'relevance_judge_decision':
                this.handleRelevanceJudgeDecision(event.data);
                break;
            case 'amendment_complete':
                this.handleAmendmentComplete(event.data);
                break;
            case 'aggregator_decision':
                this.handleAggregatorDecision(event.data);
                break;
            case 'complete':
                this.handleComplete(event.data);
                break;
            case 'result':
                this.handleResult(event.data);
                break;
            case 'error':
                this.handleError(event.data);
                break;
            case 'done':
                this.handleDone();
                break;
            case 'log':
                // Console log for debugging
                console.log(`[${event.data.level}] ${event.data.message}`);
                break;
        }
    }
    
    handleStatusEvent(data) {
        this.showStatusBanner(data.message, 'processing');
    }
    
    handlePipelineStart(data) {
        const pipeline = data.pipeline;
        this.showStatusBanner(`Starting ${pipeline === 'graphrag' ? 'KG-RAG' : 'Naive-RAG'} pipeline...`, 'processing');
    }
    
    handleIterationStart(data) {
        const { pipeline, iteration, max_iterations, current_query } = data;
        
        const iterationData = {
            iteration: iteration,
            query: current_query,
            status: 'processing',
            answer: null,
            judge: null,
            modifiedQuery: null
        };
        
        if (pipeline === 'graphrag') {
            this.graphragIterations.push(iterationData);
            this.updateIterationBadge('graphrag', iteration);
        } else {
            this.naiveragIterations.push(iterationData);
            this.updateIterationBadge('naiverag', iteration);
        }
        
        this.renderIterations(pipeline);
    }
    
    handleRetrievalComplete(data) {
        const { pipeline, iteration, diagnostics, num_candidates } = data;
        const iterations = pipeline === 'graphrag' ? this.graphragIterations : this.naiveragIterations;
        const iter = iterations.find(i => i.iteration === iteration);
        if (iter) {
            iter.retrievalInfo = diagnostics || { num_candidates };
        }
    }
    
    handleAnswerGenerated(data) {
        const { pipeline, iteration, answer, query } = data;
        const iterations = pipeline === 'graphrag' ? this.graphragIterations : this.naiveragIterations;
        const iter = iterations.find(i => i.iteration === iteration);
        if (iter) {
            iter.answer = answer;
            iter.query = query;
        }
        this.renderIterations(pipeline);
    }
    
    handleJudgeDecision(data) {
        const { pipeline, iteration } = data;
        const iterations = pipeline === 'graphrag' ? this.graphragIterations : this.naiveragIterations;
        const iter = iterations.find(i => i.iteration === iteration);
        if (iter) {
            if (pipeline === 'graphrag') {
                iter.judge = {
                    acceptable: data.acceptable,
                    problems: data.problems,
                    suggestion: data.suggestion,
                    notes: data.notes
                };
            } else {
                iter.judge = {
                    decision: data.decision,
                    reasoning: data.reasoning,
                    problem: data.problem,
                    suggested_solution: data.suggested_solution
                };
            }
        }
        this.renderIterations(pipeline);
    }
    
    handleQueryModified(data) {
        const { pipeline, iteration, original_query, modified_query, rationale, notes } = data;
        const iterations = pipeline === 'graphrag' ? this.graphragIterations : this.naiveragIterations;
        const iter = iterations.find(i => i.iteration === iteration);
        if (iter) {
            iter.modifiedQuery = modified_query;
            iter.modifierRationale = rationale || notes;
        }
        this.renderIterations(pipeline);
    }
    
    handleIterationComplete(data) {
        const { pipeline, iteration, status, final } = data;
        const iterations = pipeline === 'graphrag' ? this.graphragIterations : this.naiveragIterations;
        const iter = iterations.find(i => i.iteration === iteration);
        if (iter) {
            iter.status = status;
            iter.isFinal = final;
        }
        this.renderIterations(pipeline);
    }
    
    handlePipelineComplete(data) {
        const { pipeline, iterations_used, final_answer } = data;
        this.showStatusBanner(`${pipeline === 'graphrag' ? 'KG-RAG' : 'Naive-RAG'} completed with ${iterations_used} iteration(s)`, 'processing');
    }
    
    handlePipelineError(data) {
        console.error(`Pipeline ${data.pipeline} error:`, data.error);
    }
    
    handleAmendmentStart(data) {
        this.showStatusBanner(`Processing amendments for ${data.pipeline}...`, 'processing');
    }
    
    handleUUReferencesExtracted(data) {
        const { pipeline, references } = data;
        if (!this.amendmentData[pipeline]) {
            this.amendmentData[pipeline] = {};
        }
        this.amendmentData[pipeline].uuReferences = references;
        this.renderAmendmentTab();
    }
    
    handleAmendmentChainsRetrieved(data) {
        const { pipeline, chain_results, amending_uus, amendment_info } = data;
        if (!this.amendmentData[pipeline]) {
            this.amendmentData[pipeline] = {};
        }
        this.amendmentData[pipeline].chainResults = chain_results;
        this.amendmentData[pipeline].amendingUUs = amending_uus;
        this.amendmentData[pipeline].amendmentInfo = amendment_info;
        this.renderAmendmentTab();
    }
    
    handleRelevanceJudgeDecision(data) {
        const { pipeline, is_relevant, affected_aspects, reasoning } = data;
        if (!this.amendmentData[pipeline]) {
            this.amendmentData[pipeline] = {};
        }
        this.amendmentData[pipeline].relevanceResult = {
            isRelevant: is_relevant,
            affectedAspects: affected_aspects,
            reasoning: reasoning
        };
        this.renderAmendmentTab();
    }
    
    handleAmendmentComplete(data) {
        const { pipeline, has_amendments, message, amending_chunks_used, relevance_result } = data;
        if (!this.amendmentData[pipeline]) {
            this.amendmentData[pipeline] = {};
        }
        this.amendmentData[pipeline].complete = true;
        this.amendmentData[pipeline].hasAmendments = has_amendments;
        this.amendmentData[pipeline].message = message;
        this.amendmentData[pipeline].amendingChunksUsed = amending_chunks_used;
        this.renderAmendmentTab();
    }
    
    handleAggregatorDecision(data) {
        this.aggregatorResult = {
            decision: data.decision,
            rationale: data.rationale,
            finalAnswer: data.final_answer
        };
        this.renderAggregatorTab();
    }
    
    handleComplete(data) {
        this.finalAnswer = data.final_answer;
        this.showStatusBanner('Processing complete!', 'complete');
        this.renderFinalAnswerTab();
        this.switchTab('final');
    }
    
    handleResult(data) {
        console.log('Full result:', data);
        // Store complete result for reference
        this.fullResult = data;
    }
    
    handleError(data) {
        this.showStatusBanner(`Error: ${data.message}`, 'error');
    }
    
    handleDone() {
        this.isProcessing = false;
        this.setButtonLoading(false);
        if (this.eventSource) {
            this.eventSource.close();
            this.eventSource = null;
        }
    }
    
    // UI Update Methods
    
    resetState() {
        this.graphragIterations = [];
        this.naiveragIterations = [];
        this.amendmentData = {
            graphrag: null,
            naiverag: null,
            aggregated: null
        };
        this.aggregatorResult = null;
        this.finalAnswer = null;
        
        // Reset UI
        document.getElementById('graphrag-iterations').innerHTML = '<div class="empty-state">Waiting for processing...</div>';
        document.getElementById('naiverag-iterations').innerHTML = '<div class="empty-state">Waiting for processing...</div>';
        document.getElementById('amendment-content').innerHTML = '<div class="empty-state">Waiting for amendment processing...</div>';
        document.getElementById('aggregator-content').innerHTML = '<div class="empty-state">Waiting for aggregation...</div>';
        document.getElementById('final-answer-content').innerHTML = '<div class="empty-state">Processing your question...</div>';
        
        this.updateIterationBadge('graphrag', 0);
        this.updateIterationBadge('naiverag', 0);
    }
    
    setButtonLoading(loading) {
        const btn = document.getElementById('submit-btn');
        const btnText = btn.querySelector('.btn-text');
        const btnLoading = btn.querySelector('.btn-loading');
        
        btn.disabled = loading;
        btnText.classList.toggle('hidden', loading);
        btnLoading.classList.toggle('hidden', !loading);
    }
    
    showStatusBanner(message, type) {
        const banner = document.getElementById('status-banner');
        banner.classList.remove('hidden', 'processing', 'complete', 'error');
        banner.classList.add(type);
        
        const icon = banner.querySelector('.status-icon');
        const text = banner.querySelector('.status-text');
        
        switch (type) {
            case 'processing':
                icon.textContent = '⏳';
                break;
            case 'complete':
                icon.textContent = '✅';
                break;
            case 'error':
                icon.textContent = '❌';
                break;
        }
        
        text.textContent = message;
    }
    
    showResultsSection() {
        document.getElementById('results-section').classList.remove('hidden');
    }
    
    switchTab(tabId) {
        // Update tab buttons
        document.querySelectorAll('.tab-btn').forEach(btn => {
            btn.classList.toggle('active', btn.dataset.tab === tabId);
        });
        
        // Update tab panels
        document.querySelectorAll('.tab-panel').forEach(panel => {
            panel.classList.toggle('hidden', panel.id !== `${tabId}-tab`);
            panel.classList.toggle('active', panel.id === `${tabId}-tab`);
        });
    }
    
    updateIterationBadge(pipeline, count) {
        const badge = document.getElementById(`${pipeline}-iter-badge`);
        badge.textContent = count;
    }
    
    renderIterations(pipeline) {
        const container = document.getElementById(`${pipeline}-iterations`);
        const iterations = pipeline === 'graphrag' ? this.graphragIterations : this.naiveragIterations;
        
        if (iterations.length === 0) {
            container.innerHTML = '<div class="empty-state">Waiting for processing...</div>';
            return;
        }
        
        const html = iterations.map(iter => this.renderIterationCard(iter, pipeline)).join('');
        container.innerHTML = html;
    }
    
    renderIterationCard(iter, pipeline) {
        const statusClass = this.getStatusClass(iter);
        const statusLabel = this.getStatusLabel(iter);
        
        let judgeHtml = '';
        if (iter.judge) {
            if (pipeline === 'graphrag') {
                const judgeClass = iter.judge.acceptable ? 'acceptable' : 'rejected';
                judgeHtml = `
                    <div class="iteration-section">
                        <div class="section-label">🎯 Answer Judge Decision</div>
                        <div class="section-content judge ${judgeClass}">
                            <div class="judge-verdict ${judgeClass}">
                                ${iter.judge.acceptable ? '✅ ACCEPTABLE' : '❌ NOT ACCEPTABLE'}
                            </div>
                            ${iter.judge.problems ? `
                                <div class="judge-detail">
                                    <strong>Problems:</strong> ${this.escapeHtml(iter.judge.problems)}
                                </div>
                            ` : ''}
                            ${iter.judge.suggestion ? `
                                <div class="judge-detail">
                                    <strong>Suggestion:</strong> ${this.escapeHtml(iter.judge.suggestion)}
                                </div>
                            ` : ''}
                            ${iter.judge.notes ? `
                                <div class="judge-detail">
                                    <strong>Notes:</strong> ${this.escapeHtml(iter.judge.notes)}
                                </div>
                            ` : ''}
                        </div>
                    </div>
                `;
            } else {
                const judgeClass = iter.judge.decision === 'acceptable' ? 'acceptable' : 'rejected';
                judgeHtml = `
                    <div class="iteration-section">
                        <div class="section-label">🎯 Answer Judge Decision</div>
                        <div class="section-content judge ${judgeClass}">
                            <div class="judge-verdict ${judgeClass}">
                                ${iter.judge.decision === 'acceptable' ? '✅ ACCEPTABLE' : '❌ INSUFFICIENT'}
                            </div>
                            ${iter.judge.reasoning ? `
                                <div class="judge-detail">
                                    <strong>Reasoning:</strong> ${this.escapeHtml(iter.judge.reasoning)}
                                </div>
                            ` : ''}
                            ${iter.judge.problem ? `
                                <div class="judge-detail">
                                    <strong>Problem:</strong> ${this.escapeHtml(iter.judge.problem)}
                                </div>
                            ` : ''}
                            ${iter.judge.suggested_solution ? `
                                <div class="judge-detail">
                                    <strong>Suggested Solution:</strong> ${this.escapeHtml(iter.judge.suggested_solution)}
                                </div>
                            ` : ''}
                        </div>
                    </div>
                `;
            }
        }
        
        let modifiedQueryHtml = '';
        if (iter.modifiedQuery) {
            modifiedQueryHtml = `
                <div class="iteration-section">
                    <div class="section-label">🔄 Modified Query</div>
                    <div class="section-content query modified-query">${this.escapeHtml(iter.modifiedQuery)}</div>
                    ${iter.modifierRationale ? `
                        <div class="section-content" style="margin-top: 8px; font-size: 0.85rem; color: var(--text-secondary);">
                            <strong>Rationale:</strong> ${this.escapeHtml(iter.modifierRationale)}
                        </div>
                    ` : ''}
                </div>
            `;
        }
        
        return `
            <div class="iteration-card new">
                <div class="iteration-header">
                    <div class="iteration-title">
                        <span class="pipeline-indicator ${pipeline}"></span>
                        Iteration ${iter.iteration}
                    </div>
                    <span class="iteration-status ${statusClass}">${statusLabel}</span>
                </div>
                <div class="iteration-body">
                    <div class="iteration-section">
                        <div class="section-label">❓ Query</div>
                        <div class="section-content query">${this.escapeHtml(iter.query || 'Processing...')}</div>
                    </div>
                    
                    ${iter.answer ? `
                        <div class="iteration-section">
                            <div class="section-label">💬 Generated Answer</div>
                            <div class="section-content answer">${this.escapeHtml(iter.answer)}</div>
                        </div>
                    ` : ''}
                    
                    ${judgeHtml}
                    ${modifiedQueryHtml}
                </div>
            </div>
        `;
    }
    
    getStatusClass(iter) {
        switch (iter.status) {
            case 'accepted': return 'accepted';
            case 'cap_reached': return 'continuing';
            case 'continuing': return 'rejected';
            case 'processing': return 'processing';
            default: return 'processing';
        }
    }
    
    getStatusLabel(iter) {
        switch (iter.status) {
            case 'accepted': return 'Accepted';
            case 'cap_reached': return 'Cap Reached';
            case 'continuing': return 'Continuing';
            case 'processing': return 'Processing...';
            default: return 'Processing...';
        }
    }
    
    renderAmendmentTab() {
        const container = document.getElementById('amendment-content');
        const pipelines = this.currentMode === 'before_aggregation' 
            ? ['graphrag', 'naiverag']
            : ['aggregated'];
        
        let html = '';
        
        for (const pipeline of pipelines) {
            const data = this.amendmentData[pipeline];
            if (!data) continue;
            
            const pipelineLabel = pipeline === 'graphrag' ? 'KG-RAG' 
                : pipeline === 'naiverag' ? 'Naive-RAG' 
                : 'Aggregated Answer';
            
            html += `
                <div class="amendment-section">
                    <div class="amendment-section-header">
                        <span class="pipeline-indicator ${pipeline}"></span>
                        ${pipelineLabel} - Amendment Processing
                    </div>
                    <div class="amendment-section-body">
            `;
            
            // UU References
            if (data.uuReferences && data.uuReferences.length > 0) {
                html += `
                    <div class="iteration-section">
                        <div class="section-label">📚 Extracted UU References</div>
                        <div class="uu-reference-list">
                            ${data.uuReferences.map(ref => `
                                <span class="uu-reference-tag">
                                    UU No. ${ref.number} Tahun ${ref.year}
                                    ${ref.context ? `<br><small>${this.escapeHtml(ref.context)}</small>` : ''}
                                </span>
                            `).join('')}
                        </div>
                    </div>
                `;
            } else if (data.uuReferences) {
                html += `
                    <div class="iteration-section">
                        <div class="section-label">📚 Extracted UU References</div>
                        <div class="section-content">No UU references found in the answer.</div>
                    </div>
                `;
            }
            
            // Amendment Chains
            if (data.chainResults && data.chainResults.length > 0) {
                html += `
                    <div class="iteration-section">
                        <div class="section-label">🔗 Amendment Chain Relationships</div>
                        ${data.chainResults.map(chain => {
                            if (!chain.has_amendments) {
                                return `
                                    <div class="amendment-chain">
                                        <div class="chain-title">${this.formatUUKey(chain.original_uu)}</div>
                                        <div style="color: var(--text-secondary); font-style: italic;">No amendments found</div>
                                    </div>
                                `;
                            }
                            return `
                                <div class="amendment-chain">
                                    <div class="chain-title">${this.formatUUKey(chain.original_uu)}</div>
                                    <div class="chain-relationships">
                                        ${(chain.amendment_info || []).map(amd => `
                                            <div class="chain-rel">
                                                <span>${this.formatUUKey(amd.from)}</span>
                                                <span class="chain-rel-arrow">→</span>
                                                <span>${this.formatRelType(amd.type)}</span>
                                                <span class="chain-rel-arrow">→</span>
                                                <span>${this.formatUUKey(amd.to)}</span>
                                            </div>
                                        `).join('')}
                                    </div>
                                </div>
                            `;
                        }).join('')}
                    </div>
                `;
            }
            
            // Relevance Judge Result
            if (data.relevanceResult) {
                const relClass = data.relevanceResult.isRelevant ? 'relevant' : 'not-relevant';
                html += `
                    <div class="iteration-section">
                        <div class="section-label">⚖️ Amendment Relevance Judge</div>
                        <div class="relevance-result ${relClass}">
                            <div class="relevance-verdict ${relClass}">
                                ${data.relevanceResult.isRelevant ? '✅ AMENDMENTS ARE RELEVANT' : '❌ AMENDMENTS NOT RELEVANT'}
                            </div>
                            <div style="margin-top: 8px;">
                                <strong>Reasoning:</strong> ${this.escapeHtml(data.relevanceResult.reasoning || 'N/A')}
                            </div>
                            ${data.relevanceResult.affectedAspects && data.relevanceResult.affectedAspects.length > 0 ? `
                                <div class="affected-aspects">
                                    <strong>Affected Aspects:</strong>
                                    ${data.relevanceResult.affectedAspects.map(aspect => `
                                        <span class="aspect-tag">${this.escapeHtml(aspect)}</span>
                                    `).join('')}
                                </div>
                            ` : ''}
                        </div>
                    </div>
                `;
            }
            
            // Completion status
            if (data.complete) {
                html += `
                    <div class="iteration-section">
                        <div class="section-label">📋 Amendment Processing Result</div>
                        <div class="section-content">
                            <strong>Status:</strong> ${data.message || 'Complete'}<br>
                            <strong>Has Amendments:</strong> ${data.hasAmendments ? 'Yes' : 'No'}<br>
                            ${data.amendingChunksUsed !== undefined ? `<strong>Amending Chunks Used:</strong> ${data.amendingChunksUsed}` : ''}
                        </div>
                    </div>
                `;
            }
            
            html += `
                    </div>
                </div>
            `;
        }
        
        if (!html) {
            html = '<div class="empty-state">Waiting for amendment processing...</div>';
        }
        
        container.innerHTML = html;
    }
    
    renderAggregatorTab() {
        const container = document.getElementById('aggregator-content');
        
        if (!this.aggregatorResult) {
            container.innerHTML = '<div class="empty-state">Waiting for aggregation...</div>';
            return;
        }
        
        const decisionDisplay = this.formatDecision(this.aggregatorResult.decision);
        
        container.innerHTML = `
            <div class="aggregator-decision-card">
                <div class="aggregator-decision-label">Decision</div>
                <div class="aggregator-decision-value">${decisionDisplay}</div>
                <div class="aggregator-rationale">${this.escapeHtml(this.aggregatorResult.rationale || '')}</div>
            </div>
        `;
    }
    
    renderFinalAnswerTab() {
        const container = document.getElementById('final-answer-content');
        
        if (!this.finalAnswer) {
            container.innerHTML = '<div class="empty-state">Processing your question...</div>';
            return;
        }
        
        container.innerHTML = `
            <div class="final-answer-label">Complete Answer</div>
            <div class="final-answer-text">${this.escapeHtml(this.finalAnswer)}</div>
        `;
    }
    
    // Utility methods
    
    escapeHtml(text) {
        if (!text) return '';
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
    
    formatUUKey(key) {
        if (!key) return 'Unknown';
        const parts = key.replace('AMD_', '').split('_');
        if (parts.length === 2) {
            return `UU No. ${parts[0]} Tahun ${parts[1]}`;
        }
        return key;
    }
    
    formatRelType(relType) {
        const mapping = {
            'AMD_DIUBAH_DENGAN': 'diubah dengan',
            'AMD_DIUBAH_SEBAGIAN_DENGAN': 'diubah sebagian dengan',
            'AMD_DICABUT_DENGAN': 'dicabut dengan',
            'AMD_DICABUT_SEBAGIAN_DENGAN': 'dicabut sebagian dengan'
        };
        return mapping[relType] || relType;
    }
    
    formatDecision(decision) {
        const mapping = {
            'choose_graphrag': '🔗 Using KG-RAG Answer',
            'choose_naiverag': '📄 Using Naive-RAG Answer',
            'merge': '🔀 Merged Both Answers'
        };
        return mapping[decision] || decision;
    }
}

// Initialize app when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    window.legalQAApp = new LegalQAApp();
});