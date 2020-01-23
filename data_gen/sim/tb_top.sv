// Testbench 
// 2019.11.15
// Young-Ha Hwang
`timescale 		1ns/10fs
//`timescale 		1ns/10ps
//`define 		IDEAL_REF_
//`define 		POSIM_
//`define 		IDEAL_DCO_
//`define		PERR

module tb_top();

	reg clk, rstb;

	// NRZ
//	prbs_gen_1b i_prbs_gen(rstb, rstb, clk, 1'b1, 1'b0, 2'b11, prbs_out);
//	real ch_in, ch_out;
//	assign ch_in = prbs_out ? 0.5 : -0.5;

	//PAM4	
	wire [1:0] prbs_out;
//	prbs_gen_2b i_prbs_gen(rstb, rstb, clk, 1'b1, 1'b0, 2'b11, prbs_out);
	prbs_gen_1b i_prbs_gen_msb(rstb, rstb, clk, 1'b1, 1'b0, 2'b00, prbs_out[1]);
	prbs_gen_1b i_prbs_gen_lsb(rstb, rstb, clk, 1'b1, 1'b0, 2'b01, prbs_out[0]);
//	prbs_gen_1b i_prbs_gen(rstb, rstb, clk, 1'b1, 1'b0, 2'b11, prbs_out);
	
`include "./param.v"
integer seed;
	real ch_in, ch_out;
	assign ch_in = (prbs_out==2'b00) ? -3 + NOISE*3*$dist_normal(seed,0,1):
		       (prbs_out==2'b01) ? -1 + NOISE*$dist_normal(seed,0,1):
		       (prbs_out==2'b10) ? 1 + NOISE*$dist_normal(seed,0,1): 3 + NOISE*3*$dist_normal(seed,0,1);
//	assign ch_in = (prbs_out==1'b0) ? -3 : 3;

	//channel loss : 10, 17, 23, 27 dB @ 5GHz	      
//	real ch_out_snr;
	channel_17dB i_ch(ch_in,ch_out);
//	assign ch_out_snr = ch_out + NOISE*ch_out*$dist_normal(seed,0,1);

	wire [5:0] adc_out;
	reg adc_clk;
	reg clk_b;
	reg clk_bb;
	reg clk_bbb;
	reg clk_bbbb;
	reg clk_1;
	reg clk_2;
	reg clk_3;
	reg clk_4;
	reg clk_5;
	//reg dac_clk;
adc_6b i_adc(1'b0, adc_clk, ch_out, adc_out);

//dac_6b i_dac(1'b0, dac_clk, adc_out, dac_out);

	initial clk <= 1'b1;
		
//	initial adc_clk<=1'b1;

	//data export
	integer f_in, f_out;	
	initial f_in=$fopen("./data_in.txt", "w"); 
	initial f_out=$fopen("./data_out.txt","w");


parameter CLK_PERIOD = 0.05;
parameter TIMESTEP = CLK_PERIOD/20;
parameter DELAY = 7;
//`include "./param.v"
//`include "./reg.v"
parameter SIM_TIME=CLK_PERIOD*1000004;
parameter jitter = 0.001;
//	initial begin
//		prbs_out <= 1'b0;
//		#(20)
//		#(CLK_PERIOD) prbs_out <= 1'b1;
//		#(CLK_PERIOD) prbs_out <= 1'b0;
//	end
		

	always #(CLK_PERIOD/2) clk<=~clk;
	always #(CLK_PERIOD/4) clk_b<=clk;
//	always #(CLK_PERIOD/4) clk_bb<=clk_b;
//	always #(CLK_PERIOD/4) clk_bbb<=clk_bb;
//	always #(CLK_PERIOD/4) clk_1<=clk_bbb;
//	always #(CLK_PERIOD/4) clk_2<=clk_1;
//	always #(CLK_PERIOD/4) clk_3<=clk_2;
//	always #(CLK_PERIOD/4) clk_4<=clk_3;
//	always #(0.007) clk_bbbb<=clk;
	always @(posedge clk_b or negedge clk_b) #(0.0029) clk_4<=clk_b;
	always @(posedge clk_4 or negedge clk_4) #(DELAY*TIMESTEP) adc_clk<=clk_4;
//	always @(posedge clk_5 or negedge clk_5) #(DELAY*TIMESTEP) adc_clk<=clk_5;
//	always @(posedge adc_clk or negedge adc_clk) #(DELAY) dac_clk<=adc_clk;
//	assign #(DELAY) adc_clk=clk;

	// jittery clock
//	assign #(DELAY+jitter*$dist_normal(seed,0,1)) adc_clk = clk;
//	always #(DELAY+jitter*$dist_normal(seed,0,1)) adc_clk<=clk;

	always #CLK_PERIOD $fwrite(f_in, "%b\n", prbs_out); 
	always #CLK_PERIOD $fwrite(f_out, "%b\n", adc_out);






initial
begin

		rstb=1'b1;
		#0.01 
		rstb=1'b0;
		#0.01
		rstb=1'b1;


end

// Simulation start and finish
initial
begin
		$shm_open		("top.shm");
		$shm_probe		("AC");	 	// probe all
		//$shm_probe		(tb_top.i_adpll_dig_top.ssc_frac);	 	
		#(SIM_TIME)
		$finish(2);
		$shm_close();	
end


endmodule
