using System.Collections.Generic;
using MessagePipe;

namespace PoRacer.Tests
{
    internal sealed class FakePublisher<T> : IPublisher<T>
    {
        public readonly List<T> Published = new();

        public void Publish(T message) => Published.Add(message);
    }
}
