//////////////////////////////
//	ADC Modeling (6bit)		//
//	Author : GSJEONG		//
//	Update : 2018.07.15		//
//////////////////////////////

`timescale 1ns/1fs
module adc_6b(
	input	wire		pd,		// Power down
	input	wire		clk,	// Clock
	input	real		in,		// Analog in  
	output	reg	[5:0]	out		// ADC out
	);

	real	vh = 3-5*6/63.0;
	real	vl = -3+5*6/63.0;
	real	th_step;
	real	th1,th2,th3,th4,th5,th6,th7,th8,th9,th10,th11,th12,th13,th14,th15,th16,th17,th18,th19,th20,th21,th22,th23,th24,th25,th26,th27,th28,th29,th30,th31,th32,th33,th34,th35,th36,th37,th38,th39,th40,th41,th42,th43,th44,th45,th46,th47,th48,th49,th50,th51,th52,th53,th54,th55,th56,th57,th58,th59,th60,th61,th62,th63;

	assign 	th_step = (vh - vl)/64.0;
	assign	th1 = vl+th_step*1;
	assign	th2 = vl+th_step*2;
	assign	th3 = vl+th_step*3;
	assign	th4 = vl+th_step*4;
	assign	th5 = vl+th_step*5;
	assign	th6 = vl+th_step*6;
	assign	th7 = vl+th_step*7;
	assign	th8 = vl+th_step*8;
	assign	th9 = vl+th_step*9;
	assign	th10 = vl+th_step*10;
	assign	th11 = vl+th_step*11;
	assign	th12 = vl+th_step*12;
	assign	th13 = vl+th_step*13;
	assign	th14 = vl+th_step*14;
	assign	th15 = vl+th_step*15;
	assign	th16 = vl+th_step*16;
	assign	th17 = vl+th_step*17;
	assign	th18 = vl+th_step*18;
	assign	th19 = vl+th_step*19;
	assign	th20 = vl+th_step*20;
	assign	th21 = vl+th_step*21;
	assign	th22 = vl+th_step*22;
	assign	th23 = vl+th_step*23;
	assign	th24 = vl+th_step*24;
	assign	th25 = vl+th_step*25;
	assign	th26 = vl+th_step*26;
	assign	th27 = vl+th_step*27;
	assign	th28 = vl+th_step*28;
	assign	th29 = vl+th_step*29;
	assign	th30 = vl+th_step*30;
	assign	th31 = vl+th_step*31;
	assign	th32 = vl+th_step*32;
	assign	th33 = vl+th_step*33;
	assign	th34 = vl+th_step*34;
	assign	th35 = vl+th_step*35;
	assign	th36 = vl+th_step*36;
	assign	th37 = vl+th_step*37;
	assign	th38 = vl+th_step*38;
	assign	th39 = vl+th_step*39;
	assign	th40 = vl+th_step*40;
	assign	th41 = vl+th_step*41;
	assign	th42 = vl+th_step*42;
	assign	th43 = vl+th_step*43;
	assign	th44 = vl+th_step*44;
	assign	th45 = vl+th_step*45;
	assign	th46 = vl+th_step*46;
	assign	th47 = vl+th_step*47;
	assign	th48 = vl+th_step*48;
	assign	th49 = vl+th_step*49;
	assign	th50 = vl+th_step*50;
	assign	th51 = vl+th_step*51;
	assign	th52 = vl+th_step*52;
	assign	th53 = vl+th_step*53;
	assign	th54 = vl+th_step*54;
	assign	th55 = vl+th_step*55;
	assign	th56 = vl+th_step*56;
	assign	th57 = vl+th_step*57;
	assign	th58 = vl+th_step*58;
	assign	th59 = vl+th_step*59;
	assign	th60 = vl+th_step*60;
	assign	th61 = vl+th_step*61;
	assign	th62 = vl+th_step*62;
	assign	th63 = vl+th_step*63;

	always @(posedge clk or posedge pd)
		begin
		if(pd)
						out <= 6'b0;
		else
			begin
			if(in < th1)			out <= 6'b00_0000;
			else if(in < th2)		out <= 6'b00_0001;
			else if(in < th3)		out <= 6'b00_0010;
			else if(in < th4)		out <= 6'b00_0011;
			else if(in < th5)		out <= 6'b00_0100;
			else if(in < th6)		out <= 6'b00_0101;
			else if(in < th7)		out <= 6'b00_0110;
			else if(in < th8)		out <= 6'b00_0111;
			else if(in < th9)		out <= 6'b00_1000;
			else if(in < th10)		out <= 6'b00_1001;
			else if(in < th11)		out <= 6'b00_1010;
			else if(in < th12)		out <= 6'b00_1011;
			else if(in < th13)		out <= 6'b00_1100;
			else if(in < th14)		out <= 6'b00_1101;
			else if(in < th15)		out <= 6'b00_1110;
			else if(in < th16)		out <= 6'b00_1111;
			else if(in < th17)		out <= 6'b01_0000; 
			else if(in < th18)		out <= 6'b01_0001;
			else if(in < th19)		out <= 6'b01_0010;
			else if(in < th20)		out <= 6'b01_0011;
			else if(in < th21)		out <= 6'b01_0100;
			else if(in < th22)		out <= 6'b01_0101;
			else if(in < th23)		out <= 6'b01_0110;
			else if(in < th24)		out <= 6'b01_0111;
			else if(in < th25)		out <= 6'b01_1000;
			else if(in < th26)		out <= 6'b01_1001;
			else if(in < th27)		out <= 6'b01_1010;
			else if(in < th28)		out <= 6'b01_1011;
			else if(in < th29)		out <= 6'b01_1100;
			else if(in < th30)		out <= 6'b01_1101;
			else if(in < th31)		out <= 6'b01_1110;
			else if(in < th32)		out <= 6'b01_1111;
			else if(in < th33)		out <= 6'b10_0000; 
			else if(in < th34)		out <= 6'b10_0001;
			else if(in < th35)		out <= 6'b10_0010;
			else if(in < th36)		out <= 6'b10_0011;
			else if(in < th37)		out <= 6'b10_0100;
			else if(in < th38)		out <= 6'b10_0101;
			else if(in < th39)		out <= 6'b10_0110;
			else if(in < th40)		out <= 6'b10_0111;
			else if(in < th41)		out <= 6'b10_1000;
			else if(in < th42)		out <= 6'b10_1001;
			else if(in < th43)		out <= 6'b10_1010;
			else if(in < th44)		out <= 6'b10_1011;
			else if(in < th45)		out <= 6'b10_1100;
			else if(in < th46)		out <= 6'b10_1101;
			else if(in < th47)		out <= 6'b10_1110;
			else if(in < th48)		out <= 6'b10_1111;
			else if(in < th49)		out <= 6'b11_0000;
			else if(in < th50)		out <= 6'b11_0001;
			else if(in < th51)		out <= 6'b11_0010;
			else if(in < th52)		out <= 6'b11_0011;
			else if(in < th53)		out <= 6'b11_0100;
			else if(in < th54)		out <= 6'b11_0101;
			else if(in < th55)		out <= 6'b11_0110;
			else if(in < th56)		out <= 6'b11_0111;
			else if(in < th57)		out <= 6'b11_1000;
			else if(in < th58)		out <= 6'b11_1001;
			else if(in < th59)		out <= 6'b11_1010;
			else if(in < th60)		out <= 6'b11_1011;
			else if(in < th61)		out <= 6'b11_1100;
			else if(in < th62)		out <= 6'b11_1101;
			else if(in < th63)		out <= 6'b11_1110;
			else 					out <= 6'b11_1111;
			end
		end

endmodule
