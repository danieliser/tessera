<?php
namespace App\Analytics;

use App\Events\BaseEvent;

class Tracker {
    private array $events = [];

    public function track(BaseEvent $event): void {
        $this->events[] = $event;
        $this->flush();
    }

    private function flush(): void {
        foreach ($this->events as $event) {
            $event->dispatch();
        }
    }
}

class ClickTracker extends Tracker {
    public function trackClick(int $x, int $y): void {
        $event = new ClickEvent($x, $y);
        $this->track($event);
    }
}

function create_tracker(): Tracker {
    $tracker = new Tracker();
    return $tracker;
}

add_action('init', 'App\Analytics\create_tracker');
add_filter('analytics_enabled', function($enabled) {
    return true;
});
