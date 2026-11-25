# PTMR Pulse Timer

## Overview
The PTMR is a 16-bit pulse timer with one compare channel and one capture channel.
It counts edges of the selected clock and can generate a pulse train on the PTO pin.
All registers are 32 bits wide and reset to zero unless stated otherwise.

## Register Map
| Offset | Register | Name | Access | Reset |
| --- | --- | --- | --- | --- |
| 0x00 | Control Register | PTMR_CR | Write-only | - |
| 0x04 | Mode Register | PTMR_MR | Read/Write | 0x0000_0000 |
| 0x08 | Counter Value | PTMR_CV | Read-only | 0x0000_0000 |
| 0x0C | Compare Register | PTMR_RC | Read/Write | 0x0000_FFFF |
| 0x10 | Capture Register | PTMR_RA | Read-only | 0x0000_0000 |
| 0x14 | Status Register | PTMR_SR | Read-only | 0x0000_0000 |
| 0x18 | Interrupt Enable Register | PTMR_IER | Write-only | - |

## Control Register
| Bit | Name | Description |
| --- | --- | --- |
| 0 | CLKEN | Writing one enables the counter clock. |
| 1 | CLKDIS | Writing one disables the counter clock; it takes precedence over CLKEN. |
| 2 | SWTRG | Writing one resets the counter and starts it from zero. |

## Mode Register
| Bit | Name | Description |
| --- | --- | --- |
| 1:0 | TCCLKS | Clock selection: 0 = MCK/2, 1 = MCK/8, 2 = MCK/32, 3 = slow clock. |
| 2 | CPCSTOP | The counter stops when it reaches the value in PTMR_RC. |
| 3 | CPCTRG | A compare match with PTMR_RC resets the counter and restarts it. |
| 4 | WAVE | Selects waveform mode; capture is unavailable while this bit is set. |
| 5 | ETRGEDG | Capture is triggered on the rising edge of the PTI pin. |

## Status Register
| Bit | Name | Description |
| --- | --- | --- |
| 0 | COVFS | The counter overflowed since the last read of PTMR_SR. |
| 1 | CPCS | The counter matched PTMR_RC since the last read of PTMR_SR. |
| 2 | LDRAS | PTMR_RA was loaded by a capture since the last read of PTMR_SR. |
| 3 | CLKSTA | The counter clock is currently enabled. |

## Operation

### Initialization
Firmware selects the counter clock by programming TCCLKS before enabling the timer.
To start counting, firmware writes CLKEN and then SWTRG in the control register.
The CLKSTA flag confirms that the counter clock is running.

### Periodic interrupts
To obtain a periodic event, firmware writes the period to PTMR_RC and sets CPCTRG.
Each compare match then raises CPCS, which can interrupt the processor.

### Single pulse
For a one-shot delay, firmware sets CPCSTOP so that the counter halts on compare.

### Pulse capture
To measure the width of an input pulse, firmware clears WAVE and selects the trigger edge with ETRGEDG.
When a capture occurs, LDRAS is set and the captured count is read from PTMR_RA.

### Overflow handling
A long measurement can overflow the 16-bit counter, which firmware detects through COVFS.

## Example
```c
PTMR->MR = PTMR_MR_TCCLKS(1) | PTMR_MR_CPCTRG;
PTMR->RC = 1000;
PTMR->CR = PTMR_CR_CLKEN | PTMR_CR_SWTRG;
```
