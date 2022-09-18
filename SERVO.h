#define MIN_PULSE_WIDTH       500	//the shortest pulse sent to a servo  
#define MAX_PULSE_WIDTH      2500	//the longest pulse sent to a servo 
#define DEFAULT_PULSE_WIDTH  1500	//default pulse width when servo is attached
#define MAX_TIMER_COUNT		40000	//the timer TOP value

//volatile uint32_t *GPIO_INPUT_EN 	= (uint32_t*)0x10012004;

class PWM{
    public:
    static void init_pwm(){
        //set_bit_high(*GPIO_INPUT_EN, 11); 	// Enable GPIO as input at pin 17 (PWM-Servo)(GPIO 11)
        printf("Initializing PWM");
    }
};
