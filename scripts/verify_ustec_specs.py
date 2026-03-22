"""Verify USTEC contract specs and pip_value on demo account."""

import MetaTrader5 as mt5
import sys

def main():
    if not mt5.initialize():
        print(f"MT5 init failed: {mt5.last_error()}")
        sys.exit(1)

    symbol = "USTEC"
    info = mt5.symbol_info(symbol)
    if info is None:
        print(f"Symbol {symbol} not found")
        mt5.shutdown()
        sys.exit(1)

    print("=" * 60)
    print(f"USTEC CONTRACT SPECIFICATIONS")
    print("=" * 60)
    print(f"  Description:        {info.description}")
    print(f"  Path:               {info.path}")
    print(f"  Currency profit:    {info.currency_profit}")
    print(f"  Currency margin:    {info.currency_margin}")
    print()
    print("-- Lot Sizing --")
    print(f"  Contract size:      {info.trade_contract_size}")
    print(f"  Min lot (volume):   {info.volume_min}")
    print(f"  Max lot (volume):   {info.volume_max}")
    print(f"  Lot step:           {info.volume_step}")
    print()
    print("-- Tick/Point Info --")
    print(f"  Point:              {info.point}")
    print(f"  Tick size:          {info.trade_tick_size}")
    print(f"  Tick value:         {info.trade_tick_value}")
    print(f"  Tick value loss:    {info.trade_tick_value_loss}")
    print(f"  Digits:             {info.digits}")
    print()
    print("-- Margin --")
    print(f"  Margin initial:     {info.margin_initial}")
    print(f"  Margin maintenance: {info.margin_maintenance}")
    print(f"  Margin hedged:      {info.margin_hedged}")
    print()
    print("-- Spread --")
    print(f"  Spread:             {info.spread}")
    print(f"  Spread float:       {info.spread_float}")

    # Calculate actual pip_value
    # For CFDs: profit per 1 point move per 1 lot = tick_value / tick_size * point
    # But more simply: tick_value tells us $ per tick_size move per 1 lot
    if info.trade_tick_size > 0:
        value_per_point = info.trade_tick_value / info.trade_tick_size * info.point
        print()
        print("-- Derived Values --")
        print(f"  $ per 1 point per 1 lot:  {value_per_point:.4f}")
        print(f"  $ per 1 point per min lot: {value_per_point * info.volume_min:.4f}")

        # The pip_value in our config should be: value_per_point
        # Because: lots = risk_amount / (stop_distance_in_points * pip_value_per_lot)
        print()
        print(f"  >>> Correct pip_value for config: {value_per_point:.4f}")
    else:
        print("  WARNING: tick_size is 0, cannot derive pip_value")

    # Sample position sizing
    print()
    print("=" * 60)
    print("SAMPLE POSITION SIZING (sanity check)")
    print("=" * 60)
    balance = 100_000
    risk_pct = 0.0175  # 1.75%
    risk_amount = balance * risk_pct
    print(f"  Balance:      ${balance:,.0f}")
    print(f"  Risk per trade: {risk_pct:.2%} = ${risk_amount:,.2f}")

    # Typical USTEC ATR(14) on H1
    for atr in [50, 75, 100, 150]:
        sl_dist = 1.5 * atr  # stop_loss_atr_multiple * ATR
        if info.trade_tick_size > 0:
            pv = info.trade_tick_value / info.trade_tick_size * info.point
        else:
            pv = 1.0  # fallback
        lots = risk_amount / (sl_dist * pv)

        # Round to lot step
        step = info.volume_step
        lots_rounded = round(round(lots / step) * step, 2)
        lots_clamped = max(info.volume_min, min(lots_rounded, info.volume_max))

        # Margin required (approximate)
        tick = mt5.symbol_info_tick(symbol)
        if tick:
            price = tick.ask if tick.ask > 0 else 20000  # fallback
        else:
            price = 20000
        margin_per_lot = price * info.trade_contract_size / 100  # 1:100 leverage

        print(f"  ATR={atr:>3}: SL={sl_dist:.0f}pts | raw={lots:.2f} lots | "
              f"rounded={lots_clamped:.2f} lots | margin~${margin_per_lot * lots_clamped:,.0f} | "
              f"risk=${sl_dist * pv * lots_clamped:,.2f}")

    # Check current tick
    tick = mt5.symbol_info_tick(symbol)
    if tick:
        print()
        print(f"  Current bid: {tick.bid}")
        print(f"  Current ask: {tick.ask}")
        if tick.bid == 0 and tick.ask == 0:
            print("  WARNING: Market appears CLOSED (bid/ask = 0)")

    mt5.shutdown()


if __name__ == "__main__":
    main()
