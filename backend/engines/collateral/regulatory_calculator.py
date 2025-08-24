"""
Regulatory Capital Calculator
============================

Comprehensive regulatory capital calculation engine supporting:
- Basel III capital requirements
- Dodd-Frank regulations
- EMIR requirements
- CFTC/SEC regulations
- Multi-jurisdiction compliance
"""

import logging
from decimal import Decimal, ROUND_HALF_UP
from typing import Dict, List, Optional, Any
from datetime import datetime, timezone
from enum import Enum

from models import (
    Portfolio, Position, AssetClass, RegulatoryCapitalRequirement
)

logger = logging.getLogger(__name__)


class RegulatoryFramework(Enum):
    """Regulatory frameworks"""
    BASEL_III = "basel_iii"
    DODD_FRANK = "dodd_frank"
    EMIR = "emir"
    CFTC = "cftc"
    SEC = "sec"
    MAS = "mas"  # Monetary Authority of Singapore
    FCA = "fca"  # Financial Conduct Authority (UK)
    ESMA = "esma"  # European Securities and Markets Authority


class RegulatoryCapitalCalculator:
    """
    Regulatory capital calculator for multi-jurisdictional compliance.
    
    Implements capital adequacy calculations for:
    - Basel III framework (banks and large institutions)
    - Dodd-Frank regulations (US systemically important institutions)
    - EMIR requirements (European derivatives)
    - Local regulatory requirements
    """
    
    def __init__(self, jurisdiction: str = "US", entity_type: str = "hedge_fund"):
        self.jurisdiction = jurisdiction
        self.entity_type = entity_type
        self.regulatory_parameters = self._initialize_regulatory_parameters()
        
    def _initialize_regulatory_parameters(self) -> Dict[str, Dict[str, Any]]:
        """Initialize regulatory parameters by framework"""
        return {
            'basel_iii': {
                'tier1_capital_ratio': Decimal('0.08'),  # 8% Tier 1 capital ratio
                'total_capital_ratio': Decimal('0.10'),   # 10% total capital ratio
                'leverage_ratio': Decimal('0.03'),        # 3% leverage ratio
                'liquidity_coverage_ratio': Decimal('1.00'),  # 100% LCR
                'risk_weight_equity': Decimal('1.00'),    # 100% risk weight for equities
                'risk_weight_bonds': Decimal('0.20'),     # 20% risk weight for government bonds
                'risk_weight_derivatives': Decimal('1.25'), # 125% risk weight for derivatives
                'counterparty_credit_adjustment': Decimal('0.05')  # 5% CCR adjustment
            },
            'dodd_frank': {
                'minimum_capital_ratio': Decimal('0.08'),  # 8% minimum capital
                'buffer_requirement': Decimal('0.025'),    # 2.5% capital conservation buffer
                'systemic_risk_buffer': Decimal('0.02'),   # 2% G-SIB buffer
                'stress_test_requirement': Decimal('0.045'), # 4.5% stress test requirement
                'volcker_rule_adjustment': Decimal('0.01'), # 1% proprietary trading adjustment
                'derivative_margin_requirement': Decimal('0.02') # 2% uncleared derivative margin
            },
            'emir': {
                'initial_margin_threshold': Decimal('50000000'),  # €50M threshold
                'variation_margin_requirement': Decimal('1.00'),   # 100% variation margin
                'clearing_requirement': True,
                'reporting_requirement': True,
                'risk_mitigation_requirement': Decimal('0.08'),   # 8% risk mitigation
                'collateral_haircut_equity': Decimal('0.15'),     # 15% haircut for equity collateral
                'collateral_haircut_bonds': Decimal('0.04')       # 4% haircut for government bonds
            },
            'sec': {
                'net_capital_requirement': Decimal('0.02'),       # 2% net capital requirement
                'customer_protection_rule': Decimal('0.15'),      # 15% customer funds segregation
                'market_risk_capital': Decimal('0.08'),           # 8% market risk capital
                'operational_risk_capital': Decimal('0.03'),      # 3% operational risk capital
                'liquidity_requirement': Decimal('0.05')          # 5% liquidity requirement
            },
            'cftc': {
                'swap_dealer_capital': Decimal('0.08'),           # 8% swap dealer capital
                'major_participant_capital': Decimal('0.10'),     # 10% major participant capital
                'initial_margin_requirement': Decimal('0.02'),    # 2% initial margin
                'variation_margin_requirement': Decimal('1.00'),  # 100% variation margin
                'liquidity_risk_management': Decimal('0.03')      # 3% liquidity risk
            }
        }
    
    async def calculate_comprehensive_regulatory_capital(self, portfolio: Portfolio) -> RegulatoryCapitalRequirement:
        """Calculate comprehensive regulatory capital requirements across all frameworks"""
        
        # Calculate requirements for each framework
        basel_iii_req = await self._calculate_basel_iii_requirement(portfolio)
        dodd_frank_req = await self._calculate_dodd_frank_requirement(portfolio)
        emir_req = await self._calculate_emir_requirement(portfolio)
        local_req = await self._calculate_local_regulatory_requirement(portfolio)
        
        # Total regulatory capital is the maximum of all requirements
        total_regulatory_capital = max(basel_iii_req, dodd_frank_req, emir_req, local_req)
        
        # Calculate capital adequacy ratio
        available_capital = portfolio.available_cash + self._calculate_eligible_collateral(portfolio)
        capital_adequacy_ratio = available_capital / total_regulatory_capital if total_regulatory_capital > 0 else Decimal('inf')
        
        return RegulatoryCapitalRequirement(
            portfolio_id=portfolio.id,
            basel_iii_requirement=basel_iii_req,
            dodd_frank_requirement=dodd_frank_req,
            emir_requirement=emir_req,
            local_regulatory_requirement=local_req,
            total_regulatory_capital=total_regulatory_capital,
            capital_adequacy_ratio=capital_adequacy_ratio,
            regulatory_framework=f"{self.jurisdiction}_{self.entity_type}",
            jurisdiction=self.jurisdiction
        )
    
    async def _calculate_basel_iii_requirement(self, portfolio: Portfolio) -> Decimal:
        """Calculate Basel III capital requirements"""
        params = self.regulatory_parameters['basel_iii']
        
        # Calculate risk-weighted assets (RWA)
        total_rwa = Decimal('0')
        
        for position in portfolio.positions:
            # Apply risk weights by asset class
            if position.asset_class == AssetClass.EQUITY:
                risk_weight = params['risk_weight_equity']
            elif position.asset_class == AssetClass.BOND:
                risk_weight = params['risk_weight_bonds']
            elif position.asset_class == AssetClass.DERIVATIVE:
                risk_weight = params['risk_weight_derivatives']
            else:
                risk_weight = Decimal('1.00')  # 100% default risk weight
            
            position_rwa = position.notional_value * risk_weight
            total_rwa += position_rwa
        
        # Apply counterparty credit risk adjustment
        ccr_adjustment = total_rwa * params['counterparty_credit_adjustment']
        adjusted_rwa = total_rwa + ccr_adjustment
        
        # Calculate minimum capital requirement (8% of RWA)
        tier1_capital_req = adjusted_rwa * params['tier1_capital_ratio']
        total_capital_req = adjusted_rwa * params['total_capital_ratio']
        
        # Apply leverage ratio constraint (3% of total exposure)
        leverage_constraint = portfolio.gross_exposure * params['leverage_ratio']
        
        # Return the higher of risk-based and leverage-based requirements
        return max(total_capital_req, leverage_constraint)
    
    async def _calculate_dodd_frank_requirement(self, portfolio: Portfolio) -> Decimal:
        """Calculate Dodd-Frank capital requirements"""
        params = self.regulatory_parameters['dodd_frank']
        
        # Base capital requirement
        total_exposure = portfolio.gross_exposure
        base_capital = total_exposure * params['minimum_capital_ratio']
        
        # Add capital conservation buffer
        conservation_buffer = total_exposure * params['buffer_requirement']
        
        # Add systemic risk buffer (if applicable)
        systemic_buffer = Decimal('0')
        if self.entity_type in ['bank', 'systemically_important']:
            systemic_buffer = total_exposure * params['systemic_risk_buffer']
        
        # Stress test capital requirement
        stress_capital = total_exposure * params['stress_test_requirement']
        
        # Volcker rule adjustment for proprietary trading
        volcker_adjustment = Decimal('0')
        proprietary_trading_exposure = self._calculate_proprietary_trading_exposure(portfolio)
        if proprietary_trading_exposure > 0:
            volcker_adjustment = proprietary_trading_exposure * params['volcker_rule_adjustment']
        
        # Derivative margin requirements
        derivative_exposure = self._calculate_derivative_exposure(portfolio)
        derivative_margin = derivative_exposure * params['derivative_margin_requirement']
        
        total_dodd_frank_req = (
            base_capital + conservation_buffer + systemic_buffer + 
            stress_capital + volcker_adjustment + derivative_margin
        )
        
        return total_dodd_frank_req
    
    async def _calculate_emir_requirement(self, portfolio: Portfolio) -> Decimal:
        """Calculate EMIR (European Market Infrastructure Regulation) requirements"""
        params = self.regulatory_parameters['emir']
        
        # EMIR primarily applies to derivatives
        derivative_positions = [pos for pos in portfolio.positions if pos.asset_class == AssetClass.DERIVATIVE]
        
        if not derivative_positions:
            return Decimal('0')
        
        total_derivative_notional = sum(pos.notional_value for pos in derivative_positions)
        
        # Initial margin requirement for uncleared derivatives
        initial_margin_req = Decimal('0')
        if total_derivative_notional > params['initial_margin_threshold']:
            # Apply initial margin calculation for above-threshold exposure
            excess_exposure = total_derivative_notional - params['initial_margin_threshold']
            initial_margin_req = excess_exposure * Decimal('0.02')  # 2% initial margin
        
        # Risk mitigation requirement
        risk_mitigation_req = total_derivative_notional * params['risk_mitigation_requirement']
        
        # Collateral requirements with haircuts
        collateral_req = self._calculate_emir_collateral_requirement(derivative_positions, params)
        
        return initial_margin_req + risk_mitigation_req + collateral_req
    
    def _calculate_emir_collateral_requirement(self, derivative_positions: List[Position], params: Dict) -> Decimal:
        """Calculate EMIR collateral requirements with haircuts"""
        total_collateral_req = Decimal('0')
        
        for position in derivative_positions:
            # Base collateral requirement
            base_collateral = position.notional_value * Decimal('0.01')  # 1% base
            
            # Apply haircuts based on collateral type
            # This would be determined by actual collateral posted
            # For now, assume worst-case scenario (equity collateral)
            haircut_adjustment = base_collateral * params['collateral_haircut_equity']
            
            total_collateral_req += base_collateral + haircut_adjustment
        
        return total_collateral_req
    
    async def _calculate_local_regulatory_requirement(self, portfolio: Portfolio) -> Decimal:
        """Calculate local regulatory requirements based on jurisdiction"""
        
        if self.jurisdiction == "US":
            return await self._calculate_us_requirements(portfolio)
        elif self.jurisdiction == "EU":
            return await self._calculate_eu_requirements(portfolio)
        elif self.jurisdiction == "UK":
            return await self._calculate_uk_requirements(portfolio)
        elif self.jurisdiction == "SG":
            return await self._calculate_singapore_requirements(portfolio)
        else:
            # Default conservative requirement
            return portfolio.gross_exposure * Decimal('0.08')  # 8% default
    
    async def _calculate_us_requirements(self, portfolio: Portfolio) -> Decimal:
        """Calculate US-specific regulatory requirements"""
        sec_params = self.regulatory_parameters['sec']
        cftc_params = self.regulatory_parameters['cftc']
        
        # SEC requirements for investment advisers
        sec_req = portfolio.gross_exposure * sec_params['net_capital_requirement']
        
        # CFTC requirements for swap activities
        swap_exposure = self._calculate_swap_exposure(portfolio)
        cftc_req = swap_exposure * cftc_params['swap_dealer_capital']
        
        return max(sec_req, cftc_req)
    
    async def _calculate_eu_requirements(self, portfolio: Portfolio) -> Decimal:
        """Calculate EU-specific regulatory requirements (AIFMD, MiFID II)"""
        
        # AIFMD requirements for alternative investment funds
        aifmd_req = portfolio.gross_exposure * Decimal('0.02')  # 2% AIFMD requirement
        
        # MiFID II requirements for investment firms
        mifid_req = portfolio.gross_exposure * Decimal('0.08')  # 8% MiFID II requirement
        
        return max(aifmd_req, mifid_req)
    
    async def _calculate_uk_requirements(self, portfolio: Portfolio) -> Decimal:
        """Calculate UK-specific regulatory requirements (FCA)"""
        
        # FCA CASS rules and capital requirements
        fca_req = portfolio.gross_exposure * Decimal('0.08')  # 8% FCA requirement
        
        # Additional buffer for Brexit-related requirements
        brexit_buffer = portfolio.gross_exposure * Decimal('0.01')  # 1% Brexit buffer
        
        return fca_req + brexit_buffer
    
    async def _calculate_singapore_requirements(self, portfolio: Portfolio) -> Decimal:
        """Calculate Singapore-specific regulatory requirements (MAS)"""
        
        # MAS requirements for fund managers
        mas_req = portfolio.gross_exposure * Decimal('0.05')  # 5% MAS requirement
        
        return mas_req
    
    def _calculate_eligible_collateral(self, portfolio: Portfolio) -> Decimal:
        """Calculate eligible collateral for regulatory capital purposes"""
        
        # Only certain assets qualify as regulatory capital
        eligible_collateral = Decimal('0')
        
        for position in portfolio.positions:
            if position.asset_class == AssetClass.BOND:
                # Government bonds typically qualify with haircuts
                if position.country in ['US', 'DE', 'UK', 'JP']:  # AAA-rated government bonds
                    eligible_collateral += position.market_value * Decimal('0.95')  # 5% haircut
            elif position.asset_class == AssetClass.EQUITY:
                # Blue-chip equities with higher haircuts
                if position.notional_value > Decimal('1000000'):  # Large cap stocks
                    eligible_collateral += position.market_value * Decimal('0.80')  # 20% haircut
        
        return eligible_collateral
    
    def _calculate_proprietary_trading_exposure(self, portfolio: Portfolio) -> Decimal:
        """Calculate exposure from proprietary trading activities"""
        
        # For hedge funds, most trading would be considered proprietary
        if self.entity_type == "hedge_fund":
            return portfolio.gross_exposure * Decimal('0.80')  # 80% proprietary trading
        elif self.entity_type == "bank":
            return portfolio.gross_exposure * Decimal('0.05')  # 5% proprietary trading (Volcker rule)
        
        return Decimal('0')
    
    def _calculate_derivative_exposure(self, portfolio: Portfolio) -> Decimal:
        """Calculate total derivative exposure"""
        derivative_exposure = Decimal('0')
        
        for position in portfolio.positions:
            if position.asset_class == AssetClass.DERIVATIVE:
                derivative_exposure += position.notional_value
        
        return derivative_exposure
    
    def _calculate_swap_exposure(self, portfolio: Portfolio) -> Decimal:
        """Calculate swap exposure for CFTC requirements"""
        
        # Assume derivatives include swaps
        return self._calculate_derivative_exposure(portfolio) * Decimal('0.60')  # 60% of derivatives are swaps
    
    async def generate_regulatory_report(self, portfolio: Portfolio) -> Dict[str, Any]:
        """Generate comprehensive regulatory compliance report"""
        
        regulatory_req = await self.calculate_comprehensive_regulatory_capital(portfolio)
        
        # Compliance status
        compliance_status = {
            'basel_iii_compliant': regulatory_req.capital_adequacy_ratio >= Decimal('1.0'),
            'dodd_frank_compliant': regulatory_req.capital_adequacy_ratio >= Decimal('1.0'),
            'emir_compliant': regulatory_req.capital_adequacy_ratio >= Decimal('1.0'),
            'overall_compliant': regulatory_req.is_compliant
        }
        
        # Recommendations
        recommendations = []
        if not regulatory_req.is_compliant:
            shortfall = regulatory_req.total_regulatory_capital - (portfolio.available_cash + self._calculate_eligible_collateral(portfolio))
            recommendations.append(f"Increase capital by ${shortfall:,.2f} to meet regulatory requirements")
        
        if regulatory_req.capital_adequacy_ratio < Decimal('1.2'):
            recommendations.append("Consider increasing capital buffer above minimum requirements")
        
        return {
            'regulatory_requirements': {
                'basel_iii': float(regulatory_req.basel_iii_requirement),
                'dodd_frank': float(regulatory_req.dodd_frank_requirement),
                'emir': float(regulatory_req.emir_requirement),
                'local_regulatory': float(regulatory_req.local_regulatory_requirement),
                'total_requirement': float(regulatory_req.total_regulatory_capital)
            },
            'capital_adequacy': {
                'ratio': float(regulatory_req.capital_adequacy_ratio),
                'minimum_required': 1.0,
                'status': 'COMPLIANT' if regulatory_req.is_compliant else 'NON_COMPLIANT'
            },
            'compliance_status': compliance_status,
            'recommendations': recommendations,
            'jurisdiction': self.jurisdiction,
            'entity_type': self.entity_type,
            'calculation_date': regulatory_req.calculated_at.isoformat()
        }