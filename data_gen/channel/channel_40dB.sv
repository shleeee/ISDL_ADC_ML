module channel_40dB(input var real x, output var real y);


`timescale 1ns/1fs

	parameter real dc_gain=0.0000000000;
	real ac_r1 = 0;
	real ac_i1 = 0;
	parameter real gain_r1 = 0.0040739702;
	parameter real gain_i1 = 0.0012492273;
	parameter real  exp_r1 = 0.9821818710;
	parameter real  exp_i1 = 0.0266764709;
	real ac_r2 = 0;
	real ac_i2 = 0;
	parameter real gain_r2 = 0.0040739702;
	parameter real gain_i2 = -0.0012492273;
	parameter real  exp_r2 = 0.9821818710;
	parameter real  exp_i2 = -0.0266764709;
	real ac_r3 = 0;
	real ac_i3 = 0;
	parameter real gain_r3 = -0.0212371554;
	parameter real gain_i3 = 0.0000000000;
	parameter real  exp_r3 = 0.9818920674;
	parameter real  exp_i3 = 0.0000000000;
	real ac_r4 = 0;
	real ac_i4 = 0;
	parameter real gain_r4 = 0.0004650427;
	parameter real gain_i4 = 0.0000000000;
	parameter real  exp_r4 = 0.9987500846;
	parameter real  exp_i4 = 0.0000000000;
	real ac_r5 = 0;
	real ac_i5 = 0;
	parameter real gain_r5 = 0.0120680592;
	parameter real gain_i5 = 0.0000000000;
	parameter real  exp_r5 = 0.9926624908;
	parameter real  exp_i5 = 0.0000000000;


	always #0.001 begin
		ac_r1 <= exp_r1*ac_r1 - exp_i1*ac_i1 + gain_r1*x;
		ac_i1 <= exp_i1*ac_r1 + exp_r1*ac_i1 + gain_i1*x;
		ac_r2 <= exp_r2*ac_r2 - exp_i2*ac_i2 + gain_r2*x;
		ac_i2 <= exp_i2*ac_r2 + exp_r2*ac_i2 + gain_i2*x;
		ac_r3 <= exp_r3*ac_r3 - exp_i3*ac_i3 + gain_r3*x;
		ac_i3 <= exp_i3*ac_r3 + exp_r3*ac_i3 + gain_i3*x;
		ac_r4 <= exp_r4*ac_r4 - exp_i4*ac_i4 + gain_r4*x;
		ac_i4 <= exp_i4*ac_r4 + exp_r4*ac_i4 + gain_i4*x;
		ac_r5 <= exp_r5*ac_r5 - exp_i5*ac_i5 + gain_r5*x;
		ac_i5 <= exp_i5*ac_r5 + exp_r5*ac_i5 + gain_i5*x;
	y <= x*dc_gain
			+ac_r1
			+ac_r2
			+ac_r3
			+ac_r4
			+ac_r5;


	end
endmodule
