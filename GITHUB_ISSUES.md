# GitHub Issues for RL Training Implementation

Copy these issues to GitHub Issues tracker. Each issue is marked with labels and priority.

---

## Epic: Phase B - RL Training Implementation

**Labels**: `epic`, `phase-b`, `rl-training`

### Description
Implement complete RL training system for AgentHands to become expert at NAT agent creation.

### Goals
- Train AgentHands agent to create complete NAT agents
- Achieve 85%+ success rate
- Deploy trained model for production use

### Related Issues
See individual issues below

---

## Issue 1: Implement SkyRL Environment for NAT Agent Creation

**Labels**: `enhancement`, `rl-training`, `high-priority`

### Description
Create the SkyRL environment (`nat_agent_env.py`) that defines how AgentHands interacts with NAT agent creation tasks during RL training.

### Tasks
- [ ] Create `openhands/integrations/nat_poc/skyrl_integration/` directory
- [ ] Implement `NATAgentCreationEnv` class inheriting from `BaseTextEnv`
- [ ] Implement `init()` method to create initial prompt
- [ ] Implement `step()` method to process agent actions
- [ ] Implement `_validate_complete_agent()` for validation
- [ ] Implement `_calculate_reward()` function with multi-component rewards
- [ ] Add validation for:
  - Tool generation (NAT compliance)
  - Integration (YAML, MCP setup)
  - Workflow (end-to-end tests)
- [ ] Write unit tests for environment
- [ ] Add docstrings and type hints

### Acceptance Criteria
- Environment follows SkyRL-Gym `BaseTextEnv` interface
- Reward function includes all components (tools, integration, workflow)
- Environment can be instantiated and run through complete episode
- All validation checks work correctly
- Tests pass

### Files to Create
- `openhands/integrations/nat_poc/skyrl_integration/nat_agent_env.py`
- `openhands/integrations/nat_poc/skyrl_integration/__init__.py`
- `tests/test_nat_agent_env.py`

### Dependencies
- None (can start immediately)

### Estimated Time
3-4 days

---

## Issue 2: Create Dataset Preparation Script

**Labels**: `enhancement`, `rl-training`, `high-priority`

### Description
Implement script to generate RL training dataset using Gemini planner. Dataset consists of NAT agent specifications.

### Tasks
- [ ] Create `prepare_dataset.py` script
- [ ] Implement function to load/generate user requests
- [ ] Implement Gemini planner integration
- [ ] Extract agent specifications from Gemini plans
- [ ] Save as parquet files (train/val split 90/10)
- [ ] Add data validation
- [ ] Add progress tracking and logging
- [ ] Add error handling for Gemini API failures
- [ ] Create example user requests library (10-20 examples)
- [ ] Add CLI arguments for num_samples, output_dir

### Acceptance Criteria
- Script successfully generates 1000+ agent specs
- Train/val split is correct (90/10)
- Parquet files are valid and loadable
- Script handles Gemini API errors gracefully
- Progress is logged to console
- Specs include all required fields

### Files to Create
- `openhands/integrations/nat_poc/skyrl_integration/prepare_dataset.py`

### Dependencies
- Gemini planner (already implemented)

### Estimated Time
2-3 days

---

## Issue 3: Register Environment with SkyRL

**Labels**: `integration`, `rl-training`, `medium-priority`

### Description
Register the NAT agent creation environment with SkyRL's environment registry so it can be used during training.

### Tasks
- [ ] Copy `nat_agent_env.py` to SkyRL-Gym repo
- [ ] Add import to `skyrl_gym/envs/__init__.py`
- [ ] Register in environment registry
- [ ] Test environment can be instantiated via `gym.make()`
- [ ] Add to SkyRL documentation
- [ ] Create example usage notebook

### Acceptance Criteria
- Environment is importable from `skyrl_gym`
- Can be instantiated via registry
- Works with SkyRL training loop
- Documentation updated

### Files to Modify
- `skyrl-gym/skyrl_gym/envs/__init__.py`
- `skyrl-gym/skyrl_gym/envs/nat_agent_creation.py` (new)
- `skyrl-gym/README.md`

### Dependencies
- Issue #1 (Environment implemented)

### Estimated Time
1-2 days

---

## Issue 4: Create SkyRL Training Configuration

**Labels**: `config`, `rl-training`, `high-priority`

### Description
Create training scripts and configuration for SkyRL to train AgentHands on NAT agent creation.

### Tasks
- [ ] Create `examples/nat_agent_creation/` directory in SkyRL
- [ ] Create `train_agenthands.sh` training script
- [ ] Configure model (Qwen2.5-Coder-32B-Instruct)
- [ ] Configure GRPO algorithm parameters
- [ ] Configure distributed training (FSDP2)
- [ ] Configure vLLM inference engine
- [ ] Set batch sizes and learning rate
- [ ] Configure checkpointing
- [ ] Configure WandB logging
- [ ] Test configuration on small scale
- [ ] Document all hyperparameters

### Acceptance Criteria
- Training script runs without errors
- All hyperparameters are documented
- Checkpointing works correctly
- WandB logging captures all metrics
- Configuration is reproducible

### Files to Create
- `SkyRL/skyrl-train/examples/nat_agent_creation/train_agenthands.sh`
- `SkyRL/skyrl-train/examples/nat_agent_creation/config.yaml` (optional)
- `SkyRL/skyrl-train/examples/nat_agent_creation/README.md`

### Dependencies
- Issue #1 (Environment implemented)
- Issue #3 (Environment registered)

### Estimated Time
2-3 days

---

## Issue 5: Implement NAT Tool Validator

**Labels**: `validation`, `rl-training`, `medium-priority`

### Description
Create validation module to check NAT tool compliance, test passage, and code quality.

### Tasks
- [ ] Create `nat_validator.py` module
- [ ] Implement NAT pattern compliance checker
- [ ] Implement pytest runner integration
- [ ] Implement type hint coverage checker
- [ ] Implement error handling checker
- [ ] Add YAML config validator
- [ ] Add MCP setup script validator
- [ ] Create comprehensive validation report format
- [ ] Add unit tests for validator
- [ ] Document validation criteria

### Acceptance Criteria
- Can validate NAT tools against pattern
- Can run pytest and parse results
- Can check code quality metrics
- Can validate YAML and MCP setup
- Returns structured validation results

### Files to Create
- `openhands/integrations/nat_poc/skyrl_integration/nat_validator.py`
- `tests/test_nat_validator.py`

### Dependencies
- None (can start immediately)

### Estimated Time
2-3 days

---

## Issue 6: Create Model Evaluation Script

**Labels**: `evaluation`, `rl-training`, `medium-priority`

### Description
Implement script to evaluate trained AgentHands model on validation set.

### Tasks
- [ ] Create `evaluate_model.py` script
- [ ] Load trained checkpoint
- [ ] Run on validation set
- [ ] Calculate success metrics
- [ ] Generate evaluation report
- [ ] Add visualization of results
- [ ] Compare to baseline (pre-training)
- [ ] Export results to JSON/CSV
- [ ] Add CLI for custom evaluation runs

### Acceptance Criteria
- Can load any checkpoint
- Evaluates on validation set correctly
- Generates comprehensive report
- Metrics match training metrics
- Results are reproducible

### Files to Create
- `openhands/integrations/nat_poc/skyrl_integration/evaluate_model.py`

### Dependencies
- Issue #1 (Environment implemented)
- Issue #4 (Training config ready)

### Estimated Time
2 days

---

## Issue 7: Create Deployment Scripts

**Labels**: `deployment`, `rl-training`, `low-priority`

### Description
Create scripts to deploy trained AgentHands model for production use.

### Tasks
- [ ] Create `deploy_trained.py` script
- [ ] Implement model loading from checkpoint
- [ ] Create wrapper class for production inference
- [ ] Add Gemini planner integration
- [ ] Create end-to-end agent creation pipeline
- [ ] Add logging and monitoring
- [ ] Create example usage scripts
- [ ] Document deployment process
- [ ] Add production inference optimizations (vLLM)

### Acceptance Criteria
- Can load trained model and run inference
- Integrates with Gemini planner
- Creates complete NAT agents
- Production-ready code quality
- Well documented

### Files to Create
- `openhands/integrations/nat_poc/skyrl_integration/deploy_trained.py`
- `openhands/integrations/nat_poc/skyrl_integration/inference_wrapper.py`
- `examples/production_deployment.py`

### Dependencies
- Issue #4 (Training completed)
- Issue #6 (Evaluation shows good results)

### Estimated Time
3-4 days

---

## Issue 8: Run Initial Training Experiment

**Labels**: `experiment`, `rl-training`, `high-priority`

### Description
Run the first full-scale RL training experiment on 8x H100 GPUs for 100 epochs.

### Tasks
- [ ] Prepare hardware (8x H100 GPUs)
- [ ] Generate training dataset (1000 examples)
- [ ] Set up WandB project
- [ ] Configure Ray cluster
- [ ] Run training for 100 epochs
- [ ] Monitor training metrics
- [ ] Save checkpoints every 10 epochs
- [ ] Track GPU utilization
- [ ] Document any issues encountered
- [ ] Save final trained model
- [ ] Generate training report

### Acceptance Criteria
- Training completes successfully
- All 100 epochs finish without crashes
- Checkpoints are saved correctly
- WandB captures all metrics
- Final model achieves >70% success rate
- Training time is <3 days

### Dependencies
- Issue #1 (Environment implemented)
- Issue #2 (Dataset prepared)
- Issue #3 (Environment registered)
- Issue #4 (Training config ready)

### Estimated Time
2-3 days (training time) + setup

---

## Issue 9: Analyze Training Results and Iterate

**Labels**: `analysis`, `rl-training`, `medium-priority`

### Description
Analyze results from initial training, identify issues, and plan improvements.

### Tasks
- [ ] Run evaluation on validation set
- [ ] Analyze failure cases
- [ ] Identify common error patterns
- [ ] Review reward function effectiveness
- [ ] Check for overfitting/underfitting
- [ ] Analyze training curves
- [ ] Generate detailed analysis report
- [ ] Propose improvements for next iteration
- [ ] Document lessons learned
- [ ] Create issue for improvements

### Acceptance Criteria
- Comprehensive analysis of training results
- Clear identification of failure modes
- Actionable recommendations for improvement
- Report is well-documented

### Dependencies
- Issue #8 (Initial training completed)

### Estimated Time
2-3 days

---

## Issue 10: Optimize Reward Function

**Labels**: `enhancement`, `rl-training`, `medium-priority`

### Description
Based on initial training results, optimize the reward function for better learning.

### Tasks
- [ ] Review current reward components
- [ ] Analyze which components drive learning
- [ ] Test alternative reward formulations
- [ ] Implement curriculum learning (optional)
- [ ] Add shaped rewards for intermediate progress
- [ ] Test on small scale
- [ ] Update documentation
- [ ] Run ablation studies

### Acceptance Criteria
- Improved success rate vs baseline
- Better sample efficiency
- Clearer learning signal
- Ablation results documented

### Dependencies
- Issue #9 (Analysis completed)

### Estimated Time
3-4 days

---

## Issue 11: Create Integration Tests

**Labels**: `testing`, `rl-training`, `medium-priority`

### Description
Create comprehensive integration tests for the entire RL training pipeline.

### Tasks
- [ ] Test dataset preparation end-to-end
- [ ] Test environment with dummy agent
- [ ] Test reward calculation
- [ ] Test validation pipeline
- [ ] Test checkpoint loading/saving
- [ ] Test inference with trained model
- [ ] Add CI/CD integration
- [ ] Document test coverage

### Acceptance Criteria
- All major components have integration tests
- Tests run in CI
- >80% code coverage
- Tests are maintainable

### Files to Create
- `tests/integration/test_rl_pipeline.py`
- `tests/integration/test_environment.py`
- `tests/integration/test_validation.py`

### Dependencies
- Issue #1-7 (Core components implemented)

### Estimated Time
3-4 days

---

## Issue 12: Write Production Documentation

**Labels**: `documentation`, `rl-training`, `low-priority`

### Description
Create comprehensive documentation for production deployment and usage.

### Tasks
- [ ] Write deployment guide
- [ ] Document API for trained model
- [ ] Create user guide for agent creation
- [ ] Add troubleshooting guide
- [ ] Create architecture diagrams
- [ ] Document monitoring and logging
- [ ] Add FAQ section
- [ ] Create video tutorials (optional)

### Acceptance Criteria
- Complete documentation for deployment
- Clear user guides
- All features documented
- Examples provided

### Files to Create
- `docs/DEPLOYMENT.md`
- `docs/API_REFERENCE.md`
- `docs/USER_GUIDE.md`
- `docs/TROUBLESHOOTING.md`

### Dependencies
- Issue #7 (Deployment scripts ready)

### Estimated Time
3-4 days

---

## Issue 13: Set Up Production Monitoring

**Labels**: `monitoring`, `production`, `low-priority`

### Description
Set up monitoring and alerting for production deployment of trained AgentHands.

### Tasks
- [ ] Set up metrics collection
- [ ] Create dashboards (Grafana/WandB)
- [ ] Add alerting for failures
- [ ] Track success rates
- [ ] Monitor latency
- [ ] Track resource usage
- [ ] Add user feedback collection
- [ ] Create monitoring documentation

### Acceptance Criteria
- Real-time monitoring dashboard
- Alerts for critical failures
- Metrics tracked over time
- Dashboard is accessible

### Dependencies
- Issue #7 (Deployment ready)

### Estimated Time
2-3 days

---

## Milestone 1: Core RL Infrastructure (Week 1-2)
- Issue #1: SkyRL Environment
- Issue #2: Dataset Preparation
- Issue #5: NAT Validator

## Milestone 2: Training Setup (Week 2-3)
- Issue #3: Register with SkyRL
- Issue #4: Training Configuration
- Issue #11: Integration Tests

## Milestone 3: Initial Training (Week 3-4)
- Issue #8: Run Training
- Issue #9: Analyze Results
- Issue #10: Optimize Rewards

## Milestone 4: Production Ready (Week 5-6)
- Issue #6: Evaluation Scripts
- Issue #7: Deployment Scripts
- Issue #12: Documentation
- Issue #13: Monitoring

---

## Labels to Create in GitHub

- `epic` - Large multi-issue initiatives
- `phase-b` - Phase B work items
- `rl-training` - RL training related
- `enhancement` - New features
- `integration` - Integration work
- `config` - Configuration changes
- `validation` - Validation related
- `evaluation` - Evaluation/testing
- `deployment` - Deployment work
- `experiment` - Training experiments
- `analysis` - Analysis work
- `testing` - Test creation
- `documentation` - Documentation
- `monitoring` - Monitoring/observability
- `production` - Production readiness
- `high-priority` - Critical path
- `medium-priority` - Important
- `low-priority` - Nice to have

---

## How to Use

1. Copy each issue section above
2. Create new issue in GitHub
3. Add appropriate labels
4. Add to project board
5. Assign to team members
6. Track progress

**Total Estimated Time**: 6-8 weeks for complete implementation
