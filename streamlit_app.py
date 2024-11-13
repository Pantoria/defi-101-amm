import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class AMMCalculator:
    def __init__(self):
        # Pool 1 initial state
        self.pool1_initial_sol = 1000
        self.pool1_initial_usdc = 220000
        self.k1 = self.pool1_initial_sol * self.pool1_initial_usdc
        
        # Pool 1 after large transaction
        self.large_trade_usdc = 55000
        self.large_trade_sol = 200
        self.pool1_after_sol = self.pool1_initial_sol - self.large_trade_sol
        self.pool1_after_usdc = self.pool1_initial_usdc + self.large_trade_usdc
        
        # Pool 2 state
        self.pool2_sol = 100
        self.pool2_usdc = 22000
        self.k2 = self.pool2_sol * self.pool2_usdc
        
        # Optimal arbitrage amount (pre-calculated)
        self.optimal_sol = 18.19

    def calculate_spot_prices(self):
        """Calculate spot prices in both pools"""
        pool1_price = self.pool1_after_usdc / self.pool1_after_sol
        pool2_price = self.pool2_usdc / self.pool2_sol
        return pool1_price, pool2_price

    def calculate_arbitrage(self, sol_amount):
        """Calculate arbitrage returns for given SOL amount"""
        # Sell SOL in Pool 1 (get USDC)
        usdc_received = self.get_usdc_for_sol(sol_amount, 
                                            self.pool1_after_sol, 
                                            self.pool1_after_usdc)
        
        # Buy SOL from Pool 2 (pay USDC)
        usdc_needed = self.get_usdc_needed_for_sol(sol_amount,
                                                  self.pool2_sol,
                                                  self.pool2_usdc)
        
        # Calculate effective prices
        sell_price = usdc_received / sol_amount if sol_amount != 0 else 0
        buy_price = usdc_needed / sol_amount if sol_amount != 0 else 0
        
        profit = usdc_received - usdc_needed
        return {
            'sol_amount': sol_amount,
            'usdc_received': usdc_received,
            'usdc_needed': usdc_needed,
            'profit': profit,
            'sell_price': sell_price,
            'buy_price': buy_price
        }

    def get_usdc_for_sol(self, sol_amount, pool_sol, pool_usdc):
        """Calculate USDC received for selling SOL"""
        new_sol = pool_sol + sol_amount
        new_usdc = self.k1 / new_sol
        usdc_received = pool_usdc - new_usdc
        return usdc_received

    def get_usdc_needed_for_sol(self, sol_amount, pool_sol, pool_usdc):
        """Calculate USDC needed to buy SOL"""
        new_sol = pool_sol - sol_amount
        new_usdc = self.k2 / new_sol
        usdc_needed = new_usdc - pool_usdc
        return usdc_needed

    def plot_pools(self, arb_amount):
        """Create plots for both pools with vertical lines at trade endpoints"""
        x1 = np.linspace(150000, 400000, 1000)
        y1 = self.k1 / x1
        
        x2 = np.linspace(15000, 40000, 1000)
        y2 = self.k2 / x2
        
        # Calculate arbitrage points
        pool1_after_arb_sol = self.pool1_after_sol + arb_amount
        pool1_after_arb_usdc = self.k1 / pool1_after_arb_sol
        
        pool2_after_arb_sol = self.pool2_sol - arb_amount
        pool2_after_arb_usdc = self.k2 / pool2_after_arb_sol
        
        fig = make_subplots(rows=2, cols=1, 
                        subplot_titles=('Pool 1: Sell SOL', 'Pool 2: Buy SOL'),
                        vertical_spacing=0.15)
        
        # Pool 1 Plot
        fig.add_trace(
            go.Scatter(x=x1, y=y1, name='Pool 1 Curve', line=dict(color='blue')),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=[self.pool1_initial_usdc], y=[self.pool1_initial_sol],
                    mode='markers', name='Initial State',
                    marker=dict(color='green', size=10)),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=[self.pool1_after_usdc], y=[self.pool1_after_sol],
                    mode='markers', name='After Large Trade',
                    marker=dict(color='orange', size=10)),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=[pool1_after_arb_usdc], y=[pool1_after_arb_sol],
                    mode='markers', name='After Arbitrage',
                    marker=dict(color='red', size=10)),
            row=1, col=1
        )
        
        # Add reversed arrow for Pool 1
        fig.add_annotation(
            x=pool1_after_arb_usdc, y=pool1_after_arb_sol,
            ax=self.pool1_after_usdc, ay=self.pool1_after_sol,
            xref='x', yref='y',
            axref='x', ayref='y',
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=2,
            arrowcolor='red'
        )

            
        # Add vertical line at final trade position for Pool 1
        fig.add_shape(
            type="line",
            x0=pool1_after_arb_usdc, x1=pool1_after_arb_usdc,
            y0=0, y1=1500,
            line=dict(color="gray", width=2, dash="dash"),
            row=1, col=1
        )
        
        # Pool 2 Plot
        fig.add_trace(
            go.Scatter(x=x2, y=y2, name='Pool 2 Curve',
                    line=dict(color='blue'), showlegend=False),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=[self.pool2_usdc], y=[self.pool2_sol],
                    mode='markers', name='Initial State',
                    marker=dict(color='green', size=10), showlegend=False),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=[pool2_after_arb_usdc], y=[pool2_after_arb_sol],
                    mode='markers', name='After Arbitrage',
                    marker=dict(color='red', size=10), showlegend=False),
            row=2, col=1
        )
        
        # Add reversed arrow for Pool 2
        fig.add_annotation(
            x=pool2_after_arb_usdc, y=pool2_after_arb_sol,
            ax=self.pool2_usdc, ay=self.pool2_sol,
            xref='x2', yref='y2',
            axref='x2', ayref='y2',
            showarrow=True,
            arrowhead=2,
            arrowsize=1,
            arrowwidth=2,
            arrowcolor='red'
        )

        
        # Add vertical line at final trade position for Pool 2
        fig.add_shape(
            type="line",
            x0=pool2_after_arb_usdc, x1=pool2_after_arb_usdc,
            y0=0, y1=150,
            line=dict(color="gray", width=2, dash="dash"),
            row=2, col=1
        )

        
        # Add USDC/SOL ratio annotations at final positions
        fig.add_annotation(
            x=pool1_after_arb_usdc,
            y=0,
            text=f"USDC: {pool1_after_arb_usdc:,.0f}",
            showarrow=False,
            yshift=-30,
            row=1, col=1
        )
        fig.add_annotation(
            x=pool1_after_arb_usdc,
            y=pool1_after_arb_sol,
            text=f"SOL: {pool1_after_arb_sol:.2f}",
            showarrow=False,
            xshift=40,
            yshift=20,
            row=1, col=1
        )
        
        fig.add_annotation(
            x=pool2_after_arb_usdc,
            y=0,
            text=f"USDC: {pool2_after_arb_usdc:,.0f}",
            showarrow=False,
            yshift=-30,
            row=2, col=1
        )
        fig.add_annotation(
            x=pool2_after_arb_usdc,
            y=pool2_after_arb_sol,
            text=f"SOL: {pool2_after_arb_sol:.2f}",
            showarrow=False,
            xshift=30,
            yshift=20,
            row=2, col=1
        )
        
        fig.update_layout(
            height=800,
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            )
        )
        
        fig.update_xaxes(title_text='USDC Reserve', row=1, col=1)
        fig.update_xaxes(title_text='USDC Reserve', row=2, col=1)
        fig.update_yaxes(title_text='SOL Reserve', row=1, col=1)
        fig.update_yaxes(title_text='SOL Reserve', row=2, col=1)
        
        return fig

def main():
    st.set_page_config(page_title="AMM Arbitrage Calculator (AMM)", layout="wide")
    
    st.title("🔄 AMM Arbitrage Calculator (AMM)")
    
    # Initialize session state for amount if not exists
    if 'sol_amount' not in st.session_state:
        st.session_state.sol_amount = 18.19  # Default to optimal amount
    
    calculator = AMMCalculator()
    
    # Calculate spot prices
    pool1_price, pool2_price = calculator.calculate_spot_prices()
    
    st.markdown(f"""
    ### Current Market State
    
    **Pool 1 (Sell Here)**
    - Initial: 1000 SOL & 220K USDC
    - After large trade: {calculator.pool1_after_sol:.2f} SOL & {calculator.pool1_after_usdc:,.2f} USDC
    - Current Price: ${pool1_price:.2f} per SOL
    
    **Pool 2 (Buy Here)**
    - Current: 100 SOL & 22K USDC
    - Current Price: ${pool2_price:.2f} per SOL
    
    **Price Difference**: ${pool1_price - pool2_price:.2f} per SOL
    """)
    
    # Create two columns
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Create input columns for amount controls
        input_col1, input_col2, input_col3 = st.columns([1, 1, 1])
        
        def set_optimal():
            st.session_state.sol_amount = calculator.optimal_sol
        
        with input_col3:
            # Button to set optimal amount (place first to avoid layout shift)
            st.button("Set Optimal Amount", on_click=set_optimal)
        
        with input_col1:
            # Numeric input for precise amount
            sol_input = st.number_input(
                "Enter SOL amount:",
                min_value=0.0,
                max_value=50.0,
                value=st.session_state.sol_amount,
                step=0.01,
                format="%.2f",
                key='number_input'
            )
        
        with input_col2:
            # Slider for visual adjustment
            sol_slider = st.slider(
                "Adjust amount:",
                min_value=0.0,
                max_value=50.0,
                value=st.session_state.sol_amount,
                step=0.01,
                key='slider'
            )
        
        # Update session state based on either input
        st.session_state.sol_amount = sol_input if sol_input != st.session_state.sol_amount else sol_slider
        
        # Use the session state value
        sol_amount = st.session_state.sol_amount
        
        st.plotly_chart(calculator.plot_pools(sol_amount), use_container_width=True)
    
    with col2:
        st.subheader("Arbitrage Instructions")
        
        result = calculator.calculate_arbitrage(sol_amount)
        
        st.markdown(f"""
        ### Step-by-Step Trade Instructions
        
        1. **Buy {sol_amount:.2f} SOL from Pool 2**
           - Pay: ${result['usdc_needed']:,.2f} USDC
           - Effective buy price: ${result['buy_price']:.2f}/SOL
        
        2. **Sell {sol_amount:.2f} SOL in Pool 1**
           - Receive: ${result['usdc_received']:,.2f} USDC
           - Effective sell price: ${result['sell_price']:.2f}/SOL
        
        3. **Profit: ${result['profit']:,.2f} USDC**
        """)
        
        # Add metrics
        metrics_col1, metrics_col2 = st.columns(2)
        with metrics_col1:
            st.metric("USDC Received", f"${result['usdc_received']:,.2f}")
            st.metric("Buy Price", f"${result['buy_price']:.2f}")
        with metrics_col2:
            st.metric("USDC Needed", f"${result['usdc_needed']:,.2f}")
            st.metric("Sell Price", f"${result['sell_price']:.2f}")
        
        st.metric("Net Profit", 
                 f"${result['profit']:,.2f}",
                 delta=f"{result['profit']:,.2f}")
        
        if abs(sol_amount - calculator.optimal_sol) < 0.01:
            st.success("🎯 This is the optimal arbitrage amount!")
        elif result['profit'] > 0:
            st.success("✅ Profitable trade!")
            if sol_amount < calculator.optimal_sol:
                st.info("💡 Tip: You can increase profit by increasing the amount")
            elif sol_amount > calculator.optimal_sol:
                st.info("💡 Tip: You can increase profit by decreasing the amount")
        else:
            st.warning("⚠️ This trade would result in a loss.")
        
        st.markdown("""
        ### How to Maximize Profit
        1. Look for price differences between pools
        2. Buy from the lower-priced pool (Pool 2)
        3. Sell in the higher-priced pool (Pool 1)
        4. The optimal amount balances the price impact in both pools
        5. Use the 'Set Optimal Amount' button to find the best trade size
        """)

if __name__ == "__main__":
    main()