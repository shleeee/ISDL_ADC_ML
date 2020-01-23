//////////////////////////////
//	DAC Modeling (6bit)		//
//	Author : SHROH			//
//	Update : 2019.11.19		//
//////////////////////////////

`timescale 1ns/1fs
module dac_6b(
	input	wire		pd,		// Power down
	input	wire		clk,	// Clock
	input	wire	[5:0]	in,		// Analog in  
	output	real		out		// ADC out
	);

	reg	[5:0]	in_reg;
	real	vh = 3;
	real	vl = -3;
	real	vswing = vh - vl;
	always @(posedge clk) begin
		in_reg <= in;
	end
	assign out = vswing*(0.5*in_reg[5]+0.25*in_reg[4]+0.125*in_reg[3]+0.0625*in_reg[2]+0.03125*in_reg[1]+0.015625*in_reg[0]);
