#include <stdint.h>
#include <stdio.h>
#include <time.h>

volatile uint32_t *RTC_OUTPUT_HIGH 	= (uint32_t*)0x1000004C; // High bits of RTC Counter
volatile uint32_t *RTC_OUTPUT_LOW 	= (uint32_t*)0x10000048; // Low bits of RTC Counter
volatile uint32_t *RTC_CONFIG		= (uint32_t*)0x10000040; // Low bits of RTC Counter

void rtc_setup() {
	*RTC_CONFIG  |= (1 << 12);
}

int save_rtc_low() {

	int rtc_low_value = *RTC_OUTPUT_LOW;
	return rtc_low_value;
}

int save_rtc_high() {
	int rtc_high_value = *RTC_OUTPUT_HIGH;
	return rtc_high_value;
}
