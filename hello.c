//custom write delay function since we do not have one like an Arduino
void delay(int number_of_microseconds){ //not actually number of seconds

// Converting time into multiples of a hundred nS
int hundred_ns = 10 * number_of_microseconds;

// Storing start time
clock_t start_time = clock();

// looping till required time is not achieved
while (clock() < start_time + hundred_ns);

}

volatile uint32_t *GPIO_INPUT_VAL 	= (uint32_t*)0x10012000;
volatile uint32_t *GPIO_INPUT_EN 	= (uint32_t*)0x10012004;
volatile uint32_t *GPIO_OUTPUT_VAL 	= (uint32_t*)0x1001200C;
volatile uint32_t *GPIO_OUTPUT_EN 	= (uint32_t*)0x10012008;
//volatile uint64_t *MTIME			= (uint64_t*)0x0200bff8;

int main() {
	*GPIO_INPUT_EN |= (1 << 9);		// Enable GPIO as input at pin 15 (Echo-Pin)(GPIO 9)
	*GPIO_INPUT_EN |= (1 << 11);		// Enable GPIO as input at pin 17 (PWM-Servo)(GPIO 11)
	*GPIO_OUTPUT_EN |= (1 << 1);		// Enable GPIO as output at pin 9 (TRIG-pin)(GPIO 1)

	while(1){

		*GPIO_OUTPUT_VAL |= (1 << 1);		// Sets pin 1 HIGH
		printf("PIN 15 ON. \n");
		*GPIO_OUTPUT_VAL &= ~(1 << 1);		// Sets pin 1 LOW
		printf("PIN 15 OFF. \n");

		while(!((*GPIO_INPUT_VAL >> 9) & 0b1));
		int echoVal = ((*GPIO_INPUT_VAL >> 9) & 0b1);
		printf("Got distance. %d \n",echoVal);

	}
}
